#!/bin/bash

# Set the namespace if needed
NAMESPACE=default

# Set the label selector to find the correct pod
LABEL_SELECTOR="app=quantexa-text-classifier"

# Find the pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l $LABEL_SELECTOR -o jsonpath="{.items[0].metadata.name}")

# Check if the pod name was found
if [ -z "$POD_NAME" ]; then
  echo "No pod found with label $LABEL_SELECTOR in namespace $NAMESPACE"
  exit 1
fi

# Set up port forwarding
echo "Setting up port forwarding for pod $POD_NAME"
kubectl port-forward -n $NAMESPACE $POD_NAME 9090:9090