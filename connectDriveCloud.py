import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Define SCOPES
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate_google_drive():
    """Authenticate and return the Google Drive service."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def get_folder_id_by_path(service, path):
    """Get the folder ID for a given path in Google Drive."""
    parts = path.strip('/').split('/')
    parent_id = 'root'  # Start from the root directory
    for part in parts:
        query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and name = '{part}'"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            raise FileNotFoundError(f"Folder '{part}' not found in path '{path}'.")
        parent_id = items[0]['id']
    return parent_id


def list_files_in_folder(service, folder_id, page_size=10):
    """List files in a specified Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=page_size,  # Adjust pageSize as needed
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])
    return items


def read_file_content(service, file_id):
    """Read the content of a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    file_io = io.BytesIO()

    downloader = MediaIoBaseDownload(file_io, request)
    done = False

    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

    file_io.seek(0)
    return file_io.read().decode('utf-8')


def main():
  # Authenticate and get the Google Drive service
  service = authenticate_google_drive()

  # Define the folder path
  folder_path = 'My Drive'

  # Get the folder ID by path
  folder_id = get_folder_id_by_path(service, folder_path)
  print(f"Folder ID for '{folder_path}': {folder_id}")

  # List files in the folder
  files = list_files_in_folder(service, folder_id)
  if not files:
    print('No files found.')
  else:
    print('Files:')
    for item in files:
      print(f"{item['name']} ({item['id']})")

  # Read the content of each file
  for item in files:
    file_id = item['id']
    file_name = item['name']
    print(f"Reading content of file: {file_name} ({file_id})")
    content = read_file_content(service, file_id)
    print(f"Content of {file_name}:")
    print(content)


if __name__ == '__main__':
  main()