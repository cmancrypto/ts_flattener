import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken
from tqdm import tqdm
import json

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text) // 4

class PythonProjectFlattener:
    def __init__(
            self,
            project_path: str,
            output_dir: str = "flattened_output",
            tokens_per_file: int = 150000,
            prioritize_paths: List[str] = None
    ):
        self.project_path = os.path.abspath(project_path)
        self.output_dir = output_dir
        self.tokens_per_file = tokens_per_file
        self.prioritize_paths = prioritize_paths or []

        print(f"🔍 Initializing Python flattener for: {self.project_path}")
        print(f"📂 Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def is_relevant_file(self, file_path: str) -> bool:
        """Check if the file should be included in the flattened output."""
        path = Path(file_path).resolve()
        normalized_path = str(path).replace(os.sep, '/')

        # Define excluded directories
        excluded_dirs = [
            '__pycache__',
            'venv',
            '.venv',
            '.env',
            'build',
            'dist',
            '.git',
            '.pytest_cache',
            'eggs',
            '.tox'
        ]

        # Check if any excluded directory is in the path
        if any(f'/{excluded}/' in normalized_path or normalized_path.endswith(f'/{excluded}')
               for excluded in excluded_dirs):
            return False

        # Only include Python files
        return normalized_path.endswith('.py')

    def get_file_priority(self, file_path: str) -> int:
        """Determine priority of a file for processing order."""
        normalized_path = str(Path(file_path)).replace(os.sep, '/')

        if any(p in normalized_path for p in self.prioritize_paths):
            return 0

        priorities = {
            'core': 1,
            'models': 2,
            'views': 3,
            'controllers': 4,
            'utils': 5,
            'tests': 6
        }

        for key, priority in priorities.items():
            if f'/{key}/' in normalized_path:
                return priority
        return 99

    def process_imports(self, content: str, file_path: str) -> str:
        """Convert relative imports to flattened format."""
        def replace_import(match):
            import_statement = match.group(0)
            import_path = match.group(2)
            if import_path.startswith('.'):
                return f"# Original import: {import_statement}\n# Flattened version would be: from {import_path} import ..."
            return import_statement

        # Match both 'from ... import ...' and 'import ...' patterns
        import_pattern = r'(from\s+|\bimport\s+)([.\w]+)([^\n]*)'
        return re.sub(import_pattern, replace_import, content)

    def flatten(self) -> Dict[str, List[str]]:
        """Flatten the project into multiple files of manageable size."""
        print("\n🔎 Scanning Python project directory...")

        all_files = []
        excluded_files = []

        for root, _, files in os.walk(self.project_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_relevant_file(file_path):
                    all_files.append(file_path)
                else:
                    excluded_files.append(file_path)

        print(f"📊 Found {len(all_files)} Python files")
        print(f"🚫 Excluded {len(excluded_files)} non-Python files")

        print("\nSample of excluded files:")
        for file in excluded_files[:5]:
            print(f"  - {os.path.relpath(file, self.project_path)}")

        print("\n📋 Sorting files by priority...")
        all_files.sort(key=self.get_file_priority)

        current_chunk = []
        current_token_count = 0
        chunk_number = 1
        chunks = {}

        print("\n⚙️ Processing Python files...")
        with tqdm(total=len(all_files), desc="Processing", unit="file") as pbar:
            for file_path in all_files:
                rel_path = os.path.relpath(file_path, self.project_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        processed_content = self.process_imports(content, file_path)
                        file_content = f"\n\n{'#' * 80}\n# File: {rel_path}\n{'#' * 80}\n\n{processed_content}"
                        file_tokens = estimate_tokens(file_content)

                        if current_token_count + file_tokens > self.tokens_per_file and current_chunk:
                            chunk_file = f"chunk_{chunk_number}.txt"
                            self.write_chunk(chunk_file, current_chunk)
                            chunks[chunk_file] = [f[0] for f in current_chunk]
                            print(f"\n💾 Wrote chunk_{chunk_number}.txt ({len(current_chunk)} files)")
                            current_chunk = []
                            current_token_count = 0
                            chunk_number += 1

                        current_chunk.append((rel_path, file_content))
                        current_token_count += file_tokens

                except Exception as e:
                    print(f"\n⚠️ Error processing {rel_path}: {str(e)}")

                pbar.update(1)

        if current_chunk:
            chunk_file = f"chunk_{chunk_number}.txt"
            self.write_chunk(chunk_file, current_chunk)
            chunks[chunk_file] = [f[0] for f in current_chunk]
            print(f"\n💾 Wrote final chunk_{chunk_number}.txt ({len(current_chunk)} files)")

        self.create_chunk_manifest(chunks)
        print(f"\n✅ Done! Created {len(chunks)} chunks")
        return chunks

    def create_chunk_manifest(self, chunks: Dict[str, List[str]]) -> str:
        """Create a manifest file describing the chunks and their contents."""
        print("📝 Creating manifest file...")
        manifest = {
            "project_path": self.project_path,
            "total_chunks": len(chunks),
            "chunks": {
                chunk_file: {
                    "files": files,
                    "file_count": len(files)
                }
                for chunk_file, files in chunks.items()
            }
        }

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def write_chunk(self, chunk_file: str, chunk_contents: List[Tuple[str, str]]):
        """Write a chunk of files to an output file."""
        output_path = os.path.join(self.output_dir, chunk_file)

        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(f"# Chunk: {chunk_file}\n\n")
            out.write("## Contained Files:\n")
            for rel_path, _ in chunk_contents:
                out.write(f"- {rel_path}\n")

            out.write("\n" + "=" * 80 + "\n\n")

            for _, content in chunk_contents:
                out.write(content)

if __name__ == "__main__":
    project_path = input("Enter Python project path: ")

    priorities = [
        "src/core",
        "src/models",
        "src/views"
    ]

    print("🚀 Starting Python Project Flattener")
    flattener = PythonProjectFlattener(
        project_path=project_path,
        prioritize_paths=priorities,
        tokens_per_file=150000
    )

    chunks = flattener.flatten()