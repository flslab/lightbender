import mimetypes
import os
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_DEFAULT_TOKEN_PATH = Path.home() / ".config" / "fls" / "token.json"


def _get_credentials(client_secret_path: str, token_path: Path) -> Credentials:
    """Load credentials from token file, refreshing or re-authenticating as needed.

    On first run this opens a browser window for the user to log in. The resulting
    token (including refresh token) is saved to token_path for subsequent runs.
    If the refresh token has been revoked, the token file is deleted and the browser
    flow is triggered again.

    Args:
        client_secret_path: Path to the OAuth client secret JSON file.
        token_path: Path where the token file is stored between runs.

    Returns:
        Valid Google OAuth2 credentials.
    """
    if token_path.exists():
        try:
            creds = cast(Credentials, Credentials.from_authorized_user_file(str(token_path), _SCOPES))
        except ValueError:
            # token file is corrupted (invalid JSON); delete and fall through to browser flow
            token_path.unlink(missing_ok=True)
        else:
            if creds.valid:
                return creds
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    token_path.write_text(creds.to_json())
                    token_path.chmod(0o600)
                    return creds
                except RefreshError:
                    token_path.unlink(missing_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, _SCOPES)
    creds = cast(Credentials, flow.run_local_server(port=0))
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    token_path.chmod(0o600)
    return creds


class DriveStorage:
    """Google Drive storage client authenticated via OAuth2."""

    def __init__(self, client_secret_path: str, root_folder_id: str, token_path: Path = _DEFAULT_TOKEN_PATH):
        """Initialise the Drive storage client and authenticate via OAuth2.

        Args:
            client_secret_path: Path to the OAuth client secret JSON file.
            root_folder_id: Drive folder ID to upload experiments into.
            token_path: Path where the OAuth token is cached between runs.
        """
        creds = _get_credentials(client_secret_path, token_path)
        self._service = build("drive", "v3", credentials=creds)
        self._root_folder_id = root_folder_id
        self._folder_cache: dict[tuple[str, str], str] = {}

    @classmethod
    def from_env(cls) -> "DriveStorage":
        """Construct a DriveStorage instance from environment variables.

        Raises:
            RuntimeError: If any required environment variable is missing.
        """
        missing = [
            var for var in ("GOOGLE_CLIENT_SECRET", "GDRIVE_FOLDER_ID")
            if not os.environ.get(var)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your credentials."
            )
        client_secret_path = os.environ["GOOGLE_CLIENT_SECRET"]
        if not Path(client_secret_path).exists():
            raise RuntimeError(
                f"Client secret file not found: {client_secret_path}\n"
                "Check the GOOGLE_CLIENT_SECRET path in your .env file."
            )
        return cls(
            client_secret_path=client_secret_path,
            root_folder_id=os.environ["GDRIVE_FOLDER_ID"],
        )

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        """Return the Drive folder ID for the given name, creating it if it does not exist.

        Args:
            name: Folder name.
            parent_id: Parent folder ID.

        Returns:
            The folder ID.
        """
        key = (parent_id, name)
        if key in self._folder_cache:
            return self._folder_cache[key]

        escaped_name = name.replace("'", "\\'")  # Drive query uses single-quoted strings; unescaped quotes break the query syntax
        query = (
            f"name='{escaped_name}' and mimeType='application/vnd.google-apps.folder' "
            f"and '{parent_id}' in parents and trashed=false"
        )
        results = self._service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if files:
            self._folder_cache[key] = files[0]["id"]
            return self._folder_cache[key]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self._service.files().create(body=metadata, fields="id").execute()
        self._folder_cache[key] = folder["id"]
        return self._folder_cache[key]

    def upload_file(self, local_path: Path, drive_path: str) -> None:
        """Upload a local file to Drive, creating intermediate folders as needed.

        Args:
            local_path: Path to the local file.
            drive_path: Destination path within the root folder, using '/' as separator
                (e.g. '2025-04-09/10-30-00_interaction/logs/fls_001.log').
        """
        parts = Path(drive_path).parts
        folder_parts, filename = parts[:-1], parts[-1]

        parent_id = self._root_folder_id
        for part in folder_parts:
            parent_id = self._get_or_create_folder(part, parent_id=parent_id)

        mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True, chunksize=5 * 1024 * 1024)
        request = self._service.files().create(
            body={"name": filename, "parents": [parent_id]},
            media_body=media,
            fields="id",
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"  Uploading {filename}... {int(status.progress() * 100)}%", end="\r", flush=True)
