#!/bin/bash
# Setup script for Databricks integration

echo "================================================"
echo "Setting up Databricks Integration"
echo "================================================"
echo ""

# Install databricks-sdk
echo "📦 Installing Databricks SDK..."
uv pip install databricks-sdk

echo ""
echo "✅ Databricks SDK installed!"
echo ""
echo "Next steps:"
echo "1. Configure Databricks authentication:"
echo "   databricks auth login"
echo ""
echo "2. Or set environment variables:"
echo "   export DATABRICKS_HOST='https://your-workspace.databricks.com'"
echo "   export DATABRICKS_TOKEN='your-token'"
echo ""
echo "3. Run the data fetch notebook on Databricks (DDA Cluster)"
echo "4. Download the data:"
echo "   python scripts/download_articles.py"
echo ""

