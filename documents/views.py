from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import DocumentUploadForm
from .models import Document
from .tasks import enqueue_processing


@login_required
def document_list(request: HttpRequest) -> HttpResponse:
    documents = Document.objects.filter(user=request.user)
    return render(
        request,
        "documents/document_list.html",
        {"documents": documents, "form": DocumentUploadForm(user=request.user)},
    )


@login_required
@require_POST
def document_upload(request: HttpRequest) -> HttpResponse:
    """HTMX endpoint: returns a fresh form, plus the new row swapped in out-of-band."""
    form = DocumentUploadForm(request.POST, request.FILES, user=request.user)
    document = None
    if form.is_valid():
        upload = form.cleaned_data["file"]
        try:
            with transaction.atomic():
                document = Document(
                    user=request.user,
                    original_name=Path(upload.name).name[:255],
                    sha256=form.sha256,
                    size_bytes=upload.size,
                    file=upload,
                )
                document.save()
                enqueue_processing(document)
        except IntegrityError:
            # Same file uploaded twice concurrently: the unique constraint decides.
            form.add_error("file", "You have already uploaded this file.")
            document = None
        if document is not None:
            form = DocumentUploadForm(user=request.user)
    return render(
        request, "documents/partials/upload_response.html", {"form": form, "document": document}
    )


@login_required
def document_row(request: HttpRequest, pk: int) -> HttpResponse:
    """Polled by HTMX while the document is processing."""
    document = get_object_or_404(Document, pk=pk, user=request.user)
    return render(request, "documents/partials/document_row.html", {"document": document})


@login_required
def document_detail(request: HttpRequest, pk: int) -> HttpResponse:
    document = get_object_or_404(Document, pk=pk, user=request.user)
    return render(
        request,
        "documents/document_detail.html",
        {"document": document, "chunks": document.chunks.all()},
    )


@login_required
@require_POST
def document_delete(request: HttpRequest, pk: int) -> HttpResponse:
    get_object_or_404(Document, pk=pk, user=request.user).delete()
    return redirect("documents:document_list")


@login_required
@require_POST
def document_reprocess(request: HttpRequest, pk: int) -> HttpResponse:
    document = get_object_or_404(Document, pk=pk, user=request.user, status=Document.Status.FAILED)
    with transaction.atomic():
        document.status = Document.Status.UPLOADED
        document.error = ""
        document.save(update_fields=["status", "error"])
        enqueue_processing(document)
    return redirect("documents:document_list")
