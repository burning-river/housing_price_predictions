#!/bin/bash
# pip install "zenml[server]==0.67.0" &&
zenml up &&
zenml artifact-store register my_artifact_store --flavor=local && 
zenml stack register new_stack -o default -a my_artifact_store &&
zenml stack set new_stack &&
zenml experiment-tracker register mlflow_tracker --flavor=mlflow &&
zenml stack update -e mlflow_tracker &&
zenml model-deployer register mlflow_deployer --flavor=mlflow &&
zenml stack update -d mlflow_deployer &&
pip install mlflow