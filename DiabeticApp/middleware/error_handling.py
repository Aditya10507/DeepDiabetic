import logging
import os
from django.shortcuts import render

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        logger.exception("Unhandled exception: %s", exception)
        show_details = os.environ.get("SHOW_ERROR_DETAILS", "").lower() == "true"
        context = {
            "error_message": "An unexpected error occurred. Please try again later.",
            "error_detail": f"{exception.__class__.__name__}: {exception}" if show_details else "",
        }
        return render(request, 'error.html', context, status=500)
