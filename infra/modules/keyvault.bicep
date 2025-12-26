@description('Key Vault name (3-24 alphanumeric or hyphen).')
param name string

param location string
param tags object = {}

@description('Use RBAC authorization (recommended).')
param enableRbacAuthorization bool = true

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: enableRbacAuthorization
    publicNetworkAccess: 'Enabled'
    softDeleteRetentionInDays: 7
    // NOTE: Purge protection can be required by some org policies; enable if needed.
    enablePurgeProtection: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
  }
}

output name string = kv.name
output id string = kv.id
output vaultUri string = kv.properties.vaultUri

