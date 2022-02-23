"""Tests for osf_upload."""
import os
import random
import string
from pathlib import Path

import pytest
import requests

import temUpload
from ioSPI.ioSPI import datasets


@pytest.fixture(autouse=True, scope="session")
def setup_teardown():
    """Test node creation and clean-up for tests."""
    # token = os.environ["TEST_TOKEN"]
    token = '0uCxTBq9DQ4e9AQFvhDTRBoeu5dt3S2nFerQ5B3WFDTzTTxuEMJQPZI1FArskePZioMWDr'
    request_headers = {"Authorization": f"Bearer {token}"}
    base_api_url = "https://api.osf.io/v2/nodes/"

    test_node_label = "test_" + "".join(
        random.choice(string.ascii_letters) for i in range(5)
    )

    print(f"Creating test node Dataset -> internal -> {test_node_label} ")

    internal_node_guid = "9jwpu"
    request_url = f"{base_api_url}{internal_node_guid}/children/"

    request_body = {
        "type": "nodes",
        "attributes": {"title": test_node_label, "category": "data"},
    }

    response = requests.post(
        request_url, headers=request_headers, json={"data": request_body}
    )
    response.raise_for_status()

    pytest.auth_token = token
    pytest.test_node_guid = response.json()["data"]["id"]
    pytest.test_node_label = test_node_label
    pytest.request_headers = request_headers
    pytest.base_api_url = base_api_url

    yield

    print(
        f"\nDeleting test node Dataset -> internal -> "
        f"{test_node_label} and its sub-components."
    )
    cleanup(pytest.test_node_guid, test_node_label)


def cleanup(node_guid, test_node_label):
    """Recursively delete nodes and subcomponents."""
    base_node_url = f"{pytest.base_api_url}{node_guid}/"

    response = requests.get(f"{base_node_url}children/", headers=pytest.request_headers)
    response.raise_for_status()

    for node_child in response.json()["data"]:
        cleanup(node_child["id"], node_child["attributes"]["title"])

    response = requests.delete(base_node_url, headers=pytest.request_headers)

    if not response.ok:
        print(
            f"Failure: {test_node_label} could not be"
            f" deleted due to error code {response.status_code}."
        )
        print(response.json()["errors"][0]["detail"])
    else:
        print(f"Success: {test_node_label} deleted")



def test_upload_dataset_from_files():
    """Test whether data is parsed from TEM wrapper and files are uploaded."""
    cwd = os.getcwd()
    test_files_path = "./test_files"
    data_file = str(Path(cwd, test_files_path, "4v6x_randomrot.star"))
    metadata_file = str(Path(cwd, test_files_path, "4v6x_randomrot.star"))
    assert temUpload.upload_dataset_from_files(pytest.auth_token,data_file,metadata_file,pytest.test_node_guid)
