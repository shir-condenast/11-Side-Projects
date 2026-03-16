#!/bin/bash

# Deploy RAG App to Databricks
# This script uploads your app to Databricks Workspace and creates a Databricks App

set -e

echo "============================================================"
echo "DEPLOYING RAG APP TO DATABRICKS"
echo "============================================================"

# Configuration
PROFILE="WORKSHOP"
WORKSPACE_PATH="/Workspace/Users/$(databricks current-user me --profile $PROFILE | jq -r .userName)/rag-interior-design"

echo ""
echo "📋 Configuration:"
echo "   Profile: $PROFILE"
echo "   Workspace Path: $WORKSPACE_PATH"
echo ""

# Step 1: Create workspace directory
echo "📁 Creating workspace directory..."
databricks workspace mkdirs "$WORKSPACE_PATH" --profile $PROFILE

# Step 2: Upload app files
echo "📤 Uploading application files..."

# Upload main app
databricks workspace import app.py "$WORKSPACE_PATH/app.py" --language PYTHON --overwrite --profile $PROFILE

# Upload source files
databricks workspace mkdirs "$WORKSPACE_PATH/src" --profile $PROFILE
databricks workspace import src/data_loader.py "$WORKSPACE_PATH/src/data_loader.py" --language PYTHON --overwrite --profile $PROFILE
databricks workspace import src/rag_loader.py "$WORKSPACE_PATH/src/rag_loader.py" --language PYTHON --overwrite --profile $PROFILE
databricks workspace import src/evaluation.py "$WORKSPACE_PATH/src/evaluation.py" --language PYTHON --overwrite --profile $PROFILE

# Upload infra files
databricks workspace mkdirs "$WORKSPACE_PATH/infra" --profile $PROFILE
databricks workspace import infra/bootstrap.py "$WORKSPACE_PATH/infra/bootstrap.py" --language PYTHON --overwrite --profile $PROFILE
databricks workspace import infra/state.py "$WORKSPACE_PATH/infra/state.py" --language PYTHON --overwrite --profile $PROFILE

# Upload requirements
databricks workspace import pyproject.toml "$WORKSPACE_PATH/pyproject.toml" --overwrite --profile $PROFILE

# Upload data
databricks workspace mkdirs "$WORKSPACE_PATH/data" --profile $PROFILE
databricks workspace import data/articles.json "$WORKSPACE_PATH/data/articles.json" --overwrite --profile $PROFILE

echo ""
echo "✅ Files uploaded successfully!"
echo ""
echo "============================================================"
echo "NEXT STEPS:"
echo "============================================================"
echo ""
echo "1. Go to your Databricks workspace:"
echo "   https://condenast-dev.cloud.databricks.com"
echo ""
echo "2. Navigate to: Apps → Create App"
echo ""
echo "3. Configure the app:"
echo "   - Name: interior-design-consultant"
echo "   - Source Code Path: $WORKSPACE_PATH"
echo "   - Entry Point: app.py"
echo "   - Compute: Select 'DDA Cluster' or create new"
echo ""
echo "4. Add environment variables (if needed):"
echo "   - OPENAI_API_KEY=your-key-here"
echo ""
echo "5. Click 'Create' and wait for deployment"
echo ""
echo "============================================================"

