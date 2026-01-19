"""
Delete FAISS Index
==================
Utility script to delete the FAISS vector store index.
This will force the system to rebuild the index from PDFs on the next run.
"""

import os
import shutil
from pathlib import Path


def delete_faiss_index(index_path: str = "./faiss_index") -> bool:
    """
    Delete the FAISS index directory and all its contents.
    
    Args:
        index_path: Path to the FAISS index directory (default: "./faiss_index")
    
    Returns:
        True if deletion was successful, False otherwise
    """
    index_dir = Path(index_path)
    
    if not index_dir.exists():
        print(f"[!] FAISS index directory not found: {index_path}")
        print("   Nothing to delete.")
        return False
    
    try:
        # List contents before deletion
        contents = list(index_dir.iterdir())
        print(f"\n[+] Found FAISS index at: {index_path}")
        print(f"   Contents: {len(contents)} items")
        for item in contents:
            print(f"      - {item.name}")
        
        # Confirm deletion
        print(f"\n[!] Deleting FAISS index directory: {index_path}")
        shutil.rmtree(index_dir)
        
        print("[+] FAISS index deleted successfully!")
        print("   The index will be rebuilt from PDFs on the next run.")
        return True
        
    except PermissionError:
        print(f"\n[!] Permission denied: Cannot delete {index_path}")
        print("   Make sure the directory is not in use by another process.")
        return False
    except Exception as e:
        print(f"\n[!] Failed to delete FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main execution function.
    """
    import sys
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "="*60)
    print("Delete FAISS Index Utility")
    print("="*60)
    
    # Get index path from environment or use default
    index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    
    print(f"\nTarget directory: {index_path}")
    
    # Ask for confirmation
    confirm = input("\n[!] Are you sure you want to delete the FAISS index? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n[!] Deletion cancelled.")
        return
    
    # Delete the index
    success = delete_faiss_index(index_path)
    
    if success:
        print("\n[+] Done! Run your RAG script again to rebuild the index from PDFs.")
    else:
        print("\n[!] Deletion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
