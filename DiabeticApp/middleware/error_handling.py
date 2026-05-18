import logging
import os
from html import escape

from django.http import HttpResponse

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
        detail_html = ""
        if show_details:
            error_detail = escape(f"{exception.__class__.__name__}: {exception}")
            detail_html = f"<code>{error_detail}</code>"

        html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DeepDiabetic Error</title>
    <style>
        body {{
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            font-family: Arial, sans-serif;
            background: #f4fafb;
            color: #12323d;
        }}
        main {{
            max-width: 680px;
            padding: 32px;
            text-align: center;
        }}
        h1 {{
            margin: 0 0 12px;
            font-size: 28px;
        }}
        p {{
            margin: 0;
            line-height: 1.5;
        }}
        code {{
            display: block;
            margin-top: 18px;
            padding: 14px;
            border-radius: 6px;
            background: #e8f3f5;
            color: #173741;
            text-align: left;
            white-space: pre-wrap;
            word-break: break-word;
        }}
    </style>
</head>
<body>
    <main>
        <h1>Something went wrong</h1>
        <p>An unexpected error occurred. Please try again later.</p>
        {detail_html}
        <p style="margin-top: 18px; font-size: 13px;">Diagnostic build: python-error-response</p>
    </main>
</body>
</html>"""
        return HttpResponse(html, status=500)
