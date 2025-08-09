"""
Data Loader Module
Handles loading and preprocessing of PDF documents
"""

import os
from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import trafilatura

class DataLoader:
    """
    Simple data loader for PDF documents
    Handles loading, chunking, and preprocessing
    """
    
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdfs"
        
        if not os.environ.get("USER_AGENT"):
            os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) DataLoader/1.0 Safari/537.36"
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2400,
            chunk_overlap=400
        )
    
    def load_pdf(self, pdf_path: str):
        """
        Load a single PDF file and return documents
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="single",
            strategy="fast"
        )
        
        try:
            docs = loader.load()
            # Enrich metadata for better citations
            file_name = Path(pdf_path).name
            for d in docs:
                d.metadata = {**(d.metadata or {}), "source": str(pdf_path), "title": file_name}
            print(f"PDF loaded: {file_name} -> {len(docs)} document(s)")
            return docs
        except Exception as e:
            print(f"Error loading PDF: {e}")
            raise
    
    def load_all_pdfs(self):
        """
        Load all PDFs from the pdfs directory
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        all_docs = []
        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")
            docs = self.load_pdf(str(pdf_file))
            all_docs.extend(docs)
        
        return all_docs
    
    def load_web_pages(self, urls):
        """
        Load HTML pages using trafilatura and return documents
        """
        all_docs = []
        for url in urls:
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    raise ValueError("failed to fetch")
                # Try to get title metadata if available
                page_title = None
                try:
                    meta = trafilatura.extract_metadata(downloaded)
                    if meta and getattr(meta, "title", None):
                        page_title = meta.title
                except Exception:
                    page_title = None
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    with_metadata=False
                )
                if not text:
                    raise ValueError("no extractable text")
                title_value = page_title or url
                doc = Document(page_content=text, metadata={"source": url, "title": title_value})
                print(f"Loaded web page: {url} -> 1 document(s)")
                all_docs.append(doc)
            except Exception as e:
                print(f"Error loading web page {url}: {e}")
        return all_docs
    
    def split_documents(self, pages):
        """
        Split documents into chunks for processing and drop tiny chunks
        """
        pages_split = self.text_splitter.split_documents(pages)
        # Drop tiny/empty chunks (<100 chars)
        filtered = [d for d in pages_split if d.page_content and len(d.page_content.strip()) >= 100]
        print(f"Created {len(filtered)} chunks from documents (filtered from {len(pages_split)})")
        return filtered

