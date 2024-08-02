import io
import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

""" 
The file token.pickle stores the user's access and refresh tokens, and is
created automatically when the authorization flow completes for the first
time. If modifying these SCOPES, delete the file token.pickle.
 """
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_FILE = 'token.pickle'
CREDENTIALS_FILE = 'credentials.json'
PAGE_SIZE_LIMIT = 100


def load_credentials():
    """Load or refresh Google Drive API credentials."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return creds


def authenticate_google_drive():
    """Authenticate and return the Google Drive service."""
    creds = load_credentials()
    service = build('drive', 'v3', credentials=creds)
    return service


def get_files(service, folder_id):
    """Retrieve files from a Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=PAGE_SIZE_LIMIT,
        fields="nextPageToken, files(id, name)"
    ).execute()
    return results.get('files', [])


def load_pickle_content(service, file_id):
    """Load pickle data from a Google Drive file."""
    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO(request.execute())
    file_content.seek(0)
    return pickle.load(file_content)




