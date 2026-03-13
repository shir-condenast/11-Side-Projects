# Implementation Checklist

Use this checklist to implement the Databricks integration step-by-step.

## ✅ Phase 1: Local Setup (5 minutes)

- [ ] **Install Databricks SDK**
  ```bash
  cd "11-Side-Projects/01 RAG For Articles"
  uv pip install databricks-sdk
  ```

- [ ] **Configure Databricks Authentication**
  ```bash
  databricks auth login
  ```
  Or add to `.env`:
  ```
  DATABRICKS_HOST=https://your-workspace.databricks.com
  DATABRICKS_TOKEN=your-token
  ```

- [ ] **Verify OpenAI API Key**
  ```bash
  # Check .env file has:
  OPENAI_API_KEY=sk-...
  ```

## ✅ Phase 2: Databricks Setup (10 minutes)

- [ ] **Upload Notebook to Databricks**
  1. Go to Databricks workspace
  2. Navigate to Workspace → Users → [your-user]
  3. Click "Import"
  4. Upload: `notebook/fetch_delta_data.py`

- [ ] **Attach to DDA Cluster**
  1. Open the notebook
  2. Click "Connect" dropdown
  3. Select: **DDA Cluster**
  4. Wait for cluster to start (if not running)

- [ ] **Run the Notebook**
  1. Click "Run All" or Cmd/Ctrl + Shift + Enter
  2. Wait for all cells to complete (~2-3 minutes)
  3. Verify output shows: "✅ Data fetch complete!"

- [ ] **Verify DBFS File**
  ```bash
  databricks fs ls dbfs:/FileStore/rag_articles/
  ```
  Should show: `articles.json`

## ✅ Phase 3: Download Data (2 minutes)

- [ ] **Run Download Script**
  ```bash
  python scripts/download_articles.py
  ```

- [ ] **Verify Local File**
  ```bash
  ls -lh data/articles.json
  ```
  Should show a file with size > 0 bytes

- [ ] **Check Data Format**
  ```bash
  head -20 data/articles.json
  ```
  Should show JSON array with objects containing: id, title, content, url

## ✅ Phase 4: Test the Application (5 minutes)

- [ ] **Start Streamlit App**
  ```bash
  streamlit run app.py
  ```

- [ ] **Test Search Functionality**
  1. App should load without errors
  2. Try a search query: "pink dining rooms"
  3. Verify results are returned
  4. Check that articles have real titles and content

- [ ] **Verify Data Source**
  1. Check article URLs start with real domain
  2. Verify content is from Architectural Digest
  3. Test multiple queries to ensure variety

## ✅ Phase 5: Validation (Optional)

- [ ] **Check Data Quality**
  ```python
  import json
  with open('data/articles.json') as f:
      articles = json.load(f)
  
  print(f"Total articles: {len(articles)}")
  print(f"Sample article: {articles[0]}")
  
  # Verify all required fields
  for article in articles:
      assert 'id' in article
      assert 'title' in article
      assert 'content' in article
      assert 'url' in article
  ```

- [ ] **Test Different Queries**
  - [ ] "modern kitchen designs"
  - [ ] "cozy bedroom ideas"
  - [ ] "minimalist living room"
  - [ ] "colorful dining spaces"

## 🎯 Success Criteria

You've successfully completed the integration when:

1. ✅ Databricks notebook runs without errors
2. ✅ DBFS file exists and contains valid JSON
3. ✅ Local `data/articles.json` has 100 articles
4. ✅ Streamlit app loads and displays results
5. ✅ Search results show real Architectural Digest articles
6. ✅ Article content is relevant and well-formatted

## 🔄 Future Iterations

Once the basic integration works, consider:

- [ ] **Increase Article Count**
  - Edit notebook: Change `LIMIT 100` to `LIMIT 500`
  - Re-run notebook and download

- [ ] **Add Date Filters**
  - Add: `AND publish_date >= '2024-01-01'`
  - Get only recent articles

- [ ] **Schedule Automatic Refreshes**
  - Create Databricks Job to run notebook daily
  - Set up cron job for download script

- [ ] **Add More Brands**
  - Modify WHERE clause: `WHERE brand IN ('Architectural Digest', 'Bon Appétit')`
  - Update UI to filter by brand

## 🐛 Troubleshooting

If something doesn't work, check:

1. **Databricks Authentication**
   ```bash
   databricks auth login
   databricks fs ls dbfs:/
   ```

2. **Cluster Status**
   - Is DDA Cluster running?
   - Check cluster logs for errors

3. **File Permissions**
   - Can you read/write to `data/` directory?
   - Check file permissions: `ls -la data/`

4. **Dependencies**
   ```bash
   uv pip list | grep databricks
   uv pip list | grep openai
   ```

## 📞 Need Help?

- Check [GETTING_STARTED.md](GETTING_STARTED.md) for detailed instructions
- Review [DATA_PIPELINE.md](DATA_PIPELINE.md) for architecture details
- See [README.md](README.md) for full documentation

---

**Estimated Total Time**: 20-25 minutes

**Last Updated**: 2024

