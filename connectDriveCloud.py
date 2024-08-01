import os
import sys
import io
import re
import pickle
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from checkResults import checkResults

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
pageSizeLimit = 100

# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
def authenticate_google_drive():
    """Authenticate and return the Google Drive service."""
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
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def get_files_id_dict(service, folder_id):
    query = f"'{folder_id}' in parents"

    results = service.files().list(
        q=query,
        pageSize=pageSizeLimit,
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    dic = {}

    if not items:
        print('No files found.')
    else:
        for item in items:
            dic[item['name']] = item['id']

        return dic


def load_and_merge_files(service, folder_id):
    query = f"'{folder_id}' in parents"

    results = service.files().list(
        q=query,
        pageSize=pageSizeLimit,  # Adjust pageSize as needed
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        merged_data = []
        for item in items:
            filename = item['name']
            file_id = item['id']

            # Retrieve the file content
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO(request.execute())

            # Load the pickle file content
            file_content.seek(0)
            try:
                data = pickle.load(file_content)
                merged_data.extend(data)
            except pickle.UnpicklingError:
                print(f'Error unpickling file: {filename}')
            except Exception as e:
                print(f'Error processing file {filename}: {str(e)}')

        return merged_data


def main():

    origin_ID = "1ScWuKuymtwcWabq_JG4OwLC18-2GUr3D"
    all_mutants_ID = "1JUgQmxD0B7nFRxN3RagUgwwDH4GNrMqB"

    service = authenticate_google_drive()
    query = f"'{origin_ID}' in parents"

    results = service.files().list(
        q=query,
        pageSize=pageSizeLimit,  # Adjust pageSize as needed
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        dic_mutant_folders = get_files_id_dict(service,all_mutants_ID)
        # Iterate through the folder
        for item in tqdm(items, desc="Checking results..."):
            filename = item['name']
            file_id = item['id']

            if filename.endswith('.pkl'):

                # Retrieve the file content
                request = service.files().get_media(fileId=file_id)
                file_content = io.BytesIO(request.execute())

                # Load the pickle file content
                file_content.seek(0)
                try:
                    # Pattern to match any of the substrings
                    pattern = r"indep_qiskit_|_output|.pkl"
                    # Remove the substrings
                    circuit_name = re.sub(pattern, "", filename)
                    oracle_pkl = pickle.load(file_content)

                    if isinstance(oracle_pkl, list):

                        mutant_folder_id = dic_mutant_folders['mutants_' + circuit_name]
                        print(circuit_name)
                        print(mutant_folder_id)
                        #mutants_path = f'{all_mutants}/mutants_{circuit_name}'
                        mutants_pkl = load_and_merge_files(service, mutant_folder_id)
                        results_df = checkResults(oracle_pkl, mutants_pkl)
                        results_df.to_csv(f'results/results_{circuit_name}.csv')

                    else:
                        print(f"Pickle file should contain a List instead of a {type(oracle_pkl)}.")
                        sys.exit(1)
                except pickle.UnpicklingError:
                    print(f'Error unpickling file: {filename}')
                except Exception as e:
                    print(f'Error processing file {filename}: {str(e)}')


if __name__ == "__main__":
    main()
