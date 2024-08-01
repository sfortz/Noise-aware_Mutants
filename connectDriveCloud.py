import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'path/to/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

# Replace 'your_folder_id_here' with the actual folder ID
folder_id = '1CXNwIIkzCBdeWnBBnAGEK2i-g_8ZSnPH'
query = f"'{folder_id}' in parents"

results = service.files().list(
    q=query,
    pageSize=10,  # Adjust pageSize as needed
    fields="nextPageToken, files(id, name)"
).execute()
items = results.get('files', [])

if not items:
    print('No files found.')
else:
    print('Files:')
    for item in items:
        print(f"{item['name']} ({item['id']})")


def read_file_content(file_id):
    request = service.files().get_media(fileId=file_id)
    file_io = io.BytesIO()

    downloader = MediaIoBaseDownload(file_io, request)
    done = False

    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

    file_io.seek(0)
    return file_io.read().decode('utf-8')


# Reading content of each file
for item in items:
    file_id = item['id']
    file_name = item['name']
    print(f"Reading content of file: {file_name} ({file_id})")
    content = read_file_content(file_id)
    print(f"Content of {file_name}:")
    print(content)
