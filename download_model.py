#!/usr/bin/env python3
"""
Script to download the model file from cloud storage (Azure or GCP).
"""
import os
import sys
import argparse
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs

def download_from_azure(blob_name: str, local_path: str):
    """Download model from Azure Blob Storage."""
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "models")
    
    if not account_name or not account_key:
        raise ValueError("Azure storage credentials not found in environment variables")
    
    print(f"Downloading from Azure Blob Storage...")
    print(f"Account: {account_name}")
    print(f"Container: {container_name}")
    print(f"Blob: {blob_name}")
    
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=account_key
    )
    
    blob_client = blob_service_client.get_blob_client(
        container=container_name, 
        blob=blob_name
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, "wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())
    
    print(f"Model downloaded successfully to: {local_path}")

def download_from_gcp(blob_name: str, local_path: str):
    """Download model from Google Cloud Storage."""
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not bucket_name:
        raise ValueError("GCP bucket name not found in environment variables")
    
    print(f"Downloading from Google Cloud Storage...")
    print(f"Bucket: {bucket_name}")
    print(f"Blob: {blob_name}")
    
    if credentials_path and os.path.exists(credentials_path):
        client = gcs.Client.from_service_account_json(credentials_path)
    else:
        client = gcs.Client()  # Use default credentials
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    blob.download_to_filename(local_path)
    print(f"Model downloaded successfully to: {local_path}")

def main():
    parser = argparse.ArgumentParser(description="Download model from cloud storage")
    parser.add_argument("--storage", choices=["azure", "gcp"], required=True,
                       help="Cloud storage provider")
    parser.add_argument("--blob-name", default="model_best_val_loss_var.pkl",
                       help="Name of the model file in cloud storage")
    parser.add_argument("--local-path", default="./pkl_file/model_best_val_loss_var.pkl",
                       help="Local path to save the model")
    
    args = parser.parse_args()
    
    try:
        if args.storage == "azure":
            download_from_azure(args.blob_name, args.local_path)
        elif args.storage == "gcp":
            download_from_gcp(args.blob_name, args.local_path)
        
        # Verify file was downloaded
        if os.path.exists(args.local_path):
            file_size = os.path.getsize(args.local_path)
            print(f"Downloaded file size: {file_size / 1024 / 1024:.2f} MB")
        else:
            print("Error: File was not downloaded successfully")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 