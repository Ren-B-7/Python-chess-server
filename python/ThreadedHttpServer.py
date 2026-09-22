"""
HTTP server with timeout capabilities and fast shutdown.

This module provides two HTTP server classes with automatic inactivity timeout
and immediate shutdown response using threading.Event instead of polling.

Classes:
    TimeoutThreadingHTTPServer: HTTP server with configurable inactivity timeout
    SSLTimeoutThreadingServer: SSL/TLS-enabled version of the timeout server

Author: Renier Barnard (renier52147@gmail.com/ renierb@axxess.co.za)
"""

from http.server import ThreadingHTTPServer
import threading
import time
import ssl
from utils.exceptions import MajorThreadedHttpServerException


class TimeoutThreadingHTTPServer(ThreadingHTTPServer):
    """
    HTTP server with configurable inactivity timeout and fast shutdown.

    This server automatically shuts down after a period of inactivity. It uses
    an event-based approach for immediate shutdown response instead of polling.

    Attributes:
        inactivity_timeout (int): Seconds of inactivity before shutdown
        stopping (bool): Flag indicating shutdown in progress
        timeout (bool): Flag indicating shutdown was triggered by timeout
        last_activity_time (float): Timestamp of last request processed

    Example:
        >>> server = TimeoutThreadingHTTPServer(
        ...     ('localhost', 8000),
        ...     MyHandler,
        ...     timeout_seconds=300
        ... )
        >>> server.serve_forever()
    """

    __slots__ = (
        "inactivity_timeout",
        "stopping",
        "timeout",
        "last_activity_time",
        "_stop_event",
        "_check_interval",
    )

    def __init__(self, server_address, handler_class, timeout_seconds=0, **kwargs):
        """
        Initialize the timeout HTTP server.

        Args:
            server_address (tuple): (host, port) tuple for server binding
            handler_class: Request handler class (e.g., BaseHTTPRequestHandler subclass)
            timeout_seconds (int): Inactivity timeout in seconds
            **kwargs: Additional arguments passed to ThreadingHTTPServer

        Raises:
            MajorThreadedHttpServerException: If server initialization fails
        """
        try:
            self._stop_event = threading.Event()
            self.stopping = False
            self.timeout = False
            self.inactivity_timeout = timeout_seconds
            self.last_activity_time = time.time()

            # Pre-calculate check interval to avoid repeated computation
            # Cap at 60 seconds to ensure responsive shutdown
            if timeout_seconds > 0:
                self._check_interval = min(60, timeout_seconds / 5)
            else:
                # Ensure that timeout_seconds is not negative
                self.inactivity_timeout = 0
                # If timeout_seconds is 0 then this wont be used
                self._check_interval = 1

            super().__init__(server_address, handler_class, **kwargs)

        except Exception as e:
            raise MajorThreadedHttpServerException(
                f"Unable to initiate server class: {e}"
            ) from e

    def server_activate(self) -> None:
        """
        Start the server and spawn the inactivity monitor thread.

        This method is called automatically during server initialization.
        It starts listening for connections and spawns a daemon thread to
        monitor for inactivity.

        Raises:
            MajorThreadedHttpServerException: If server activation fails
        """
        try:
            super().server_activate()
            self.last_activity_time = time.time()
            # Daemon thread ensures clean shutdown even if monitoring fails
            if self.inactivity_timeout > 0:
                print(f"Inactivity Timeout Set For {self.inactivity_timeout}")
                threading.Thread(target=self._monitor_inactivity, daemon=True).start()
            else:
                print("Inactivity timeout not set")
        except Exception as e:
            raise MajorThreadedHttpServerException(
                f"Unable to start server: {e}"
            ) from e

    def process_request(self, request, client_address) -> None:
        """
        Process an incoming request and update activity timestamp.

        This method intercepts each request to update the last_activity_time,
        preventing timeout while the server is actively being used.

        Args:
            request: The incoming socket request
            client_address (tuple): (host, port) of the client

        Note:
            Errors are logged but don't stop the server - it continues serving
            other requests for resilience.
        """
        try:
            # Update activity time before processing to prevent race conditions
            self.last_activity_time = time.time()
            super().process_request(request, client_address)
        except Exception as e:
            print(f"Error processing request from {client_address}: {e}")
            # Don't raise - allow server to continue serving other requests

    def _monitor_inactivity(self) -> None:
        """
        Monitor server inactivity and trigger shutdown if idle too long.

        This method runs in a separate daemon thread and uses Event.wait()
        for immediate shutdown response when requested, rather than polling
        with fixed sleep intervals.

        The monitoring loop:
        1. Calculates remaining time until timeout
        2. If timeout reached, triggers shutdown
        3. Otherwise, waits for minimum of (check_interval, remaining_time)
        4. Stop event can interrupt the wait for immediate shutdown
        """
        try:
            while not self._stop_event.is_set():
                idle_time = time.time() - self.last_activity_time

                if idle_time > self.inactivity_timeout:
                    print(
                        f"No activity for {self.inactivity_timeout / 60:.1f} minutes, "
                        f"shutting down server."
                    )
                    self.timeout = True
                    self.shutdown()
                    return

                # Wait for either timeout or explicit stop signal
                # Uses minimum to ensure we check for timeout regularly
                self._stop_event.wait(
                    timeout=min(
                        self._check_interval, self.inactivity_timeout - idle_time
                    )
                )

        except Exception as e:
            print(f"Error in inactivity monitor: {e}")

    def shutdown(self) -> None:
        """
        Gracefully shutdown the server.

        This method is idempotent - can call multiple times
        """
        if self.stopping:
            return
        self.stopping = True
        self._stop_event.set()  # Signal monitor thread to stop
        super().shutdown()

    def server_close(self) -> None:
        """
        Close the server socket and cleanup resources.

        Ensures the monitoring thread stops by setting the stop event
        before closing the server socket.
        """
        self._stop_event.set()
        self.stopping = True
        super().server_close()


class SSLTimeoutThreadingServer(TimeoutThreadingHTTPServer):
    """
    SSL/TLS-enabled HTTP server with inactivity timeout.

    This class wraps TimeoutThreadingHTTPServer with SSL/TLS support.

    Attributes:
        ssl_context (ssl.SSLContext): SSL context for secure connections

    Example:
        >>> import ssl
        >>> context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        >>> context.load_cert_chain('cert.pem', 'key.pem')
        >>> server = SSLTimeoutThreadingServer(
        ...     ('localhost', 443),
        ...     MyHandler,
        ...     timeout_seconds=300,
        ...     ssl_context=context
        ... )
    """

    def __init__(
        self, server_address, handler_class, timeout_seconds, ssl_context, **kwargs
    ):
        """
        Initialize the SSL-enabled timeout HTTP server.

        Args:
            server_address (tuple): (host, port) tuple for server binding
            handler_class: Request handler class
            timeout_seconds (int): Inactivity timeout in seconds
            ssl_context (ssl.SSLContext): Configured SSL context
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(server_address, handler_class, timeout_seconds, **kwargs)
        self.ssl_context = ssl_context

    def get_request(self):
        """
        Accept a connection and wrap it with SSL/TLS.

        This method handles the SSL/TLS handshake and recovers from common
        errors like HTTP requests sent to HTTPS ports. It loops until a
        valid connection is established or the server shuts down.

        Returns:
            tuple: (ssl_socket, client_address) for valid connections

        Note:
            Automatically recovers from SSL errors by:
            - Closing failed sockets
            - Logging the error
            - Attempting to accept the next connection
        """
        while True:
            newsocket, fromaddr = super().get_request()

            try:
                # Wrap socket with SSL/TLS
                connstream = self.ssl_context.wrap_socket(
                    newsocket,
                    server_side=True,
                )
                return connstream, fromaddr

            except ssl.SSLError as e:
                newsocket.close()

                # HTTP sent to HTTPS port is very common - not a fatal error
                if e.reason == "HTTP_REQUEST":
                    print(f"HTTP request received on HTTPS port from {fromaddr}")
                    continue

                # Log any other TLS errors and continue serving
                print(f"TLS handshake failed from {fromaddr}: {e}")
                continue

            except Exception as e:
                # Catch-all for unexpected errors during SSL handshake
                newsocket.close()
                print(f"TLS handshake failed from {fromaddr}: {e}")
