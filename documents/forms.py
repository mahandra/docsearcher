import hashlib
from pathlib import Path

from django import forms
from django.conf import settings

from .models import Document
from .parsing import EXTENSIONS, detect_kind


class DocumentUploadForm(forms.Form):
    file = forms.FileField()

    def __init__(self, *args, user, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.sha256 = ""

    def clean_file(self):
        upload = self.cleaned_data["file"]
        max_bytes = settings.DOCUMENT_MAX_UPLOAD_MB * 1024 * 1024
        if upload.size > max_bytes:
            raise forms.ValidationError(
                f"The file is larger than {settings.DOCUMENT_MAX_UPLOAD_MB} MB."
            )

        expected = EXTENSIONS.get(Path(upload.name).suffix.lower())
        if expected is None:
            raise forms.ValidationError("Only PDF and DOCX files are supported.")
        # The extension is chosen by the client; the content must agree with it.
        head = upload.read(8)
        upload.seek(0)
        if detect_kind(head) != expected:
            raise forms.ValidationError("The file content does not match its extension.")

        digest = hashlib.sha256()
        for part in upload.chunks():
            digest.update(part)
        upload.seek(0)
        self.sha256 = digest.hexdigest()
        if Document.objects.filter(user=self.user, sha256=self.sha256).exists():
            raise forms.ValidationError("You have already uploaded this file.")
        return upload
