# Azure Functions Core Tools Installation

## Automatic Installation (DevContainer)

If you're using this project with VS Code DevContainers, Azure Functions Core Tools will be automatically installed when the container is created via the postinstall script.

## Manual Installation

For manual installation instructions, see the official documentation:
[Azure Functions Core Tools - Linux Installation](https://github.com/Azure/azure-functions-core-tools/blob/v4.x/README.md#linux)

## Verification

After installation, verify it works:

```bash
func --version
```

## Configuration

### Required Application Settings

When deploying this function to Azure, you must configure the following application settings:

#### Document Intelligence Settings
```bash
# Required: Document Intelligence endpoint URL
DOC_INTEL_ENDPOINT=https://your-doc-intel-resource.cognitiveservices.azure.com/

# Optional: Document Intelligence API key (if not using Managed Identity)
DOCUMENT_INTELLIGENCE_KEY=your_api_key_here
```

#### Setting up Application Settings

**Option 1: Using Azure CLI**
```bash
# Set the endpoint (required)
az functionapp config appsettings set \
  --name <your-function-app-name> \
  --resource-group <your-resource-group> \
  --settings "DOC_INTEL_ENDPOINT=https://your-doc-intel-resource.cognitiveservices.azure.com/"

# Set the API key (optional - only if not using Managed Identity)
az functionapp config appsettings set \
  --name <your-function-app-name> \
  --resource-group <your-resource-group> \
  --settings "DOCUMENT_INTELLIGENCE_KEY=your_api_key_here"
```

**Option 2: Using Azure Portal**
1. Navigate to your Function App in the Azure Portal
2. Go to **Configuration** > **Application settings**
3. Add the required settings:
   - `DOC_INTEL_ENDPOINT`: Your Document Intelligence endpoint URL
   - `DOCUMENT_INTELLIGENCE_KEY`: Your API key (if not using Managed Identity)

**Option 3: Using Core Tools**
```bash
# Deploy with local settings (automatically uploads from local.settings.json)
func azure functionapp publish <your-function-app-name> --publish-local-settings
```

### Authentication Methods

This function supports two authentication methods (in order of preference):

1. **Managed Identity (Recommended)**: No API key required
2. **API Key**: Requires `DOCUMENT_INTELLIGENCE_KEY` setting

### Local Development

For local development, create a `local.settings.json` file:
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "DOC_INTEL_ENDPOINT": "https://your-doc-intel-resource.cognitiveservices.azure.com/",
    "DOCUMENT_INTELLIGENCE_KEY": "your_api_key_here"
  }
}
```

**⚠️ Important:** Never commit `local.settings.json` to version control as it contains sensitive information.
