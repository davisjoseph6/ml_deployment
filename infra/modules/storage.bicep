@description('Storage account name (3-24 lowercase letters+digits). Must be globally unique.')
param name string

param location string
param tags object = {}
param sku string = 'Standard_LRS'

@description('Blob containers to create.')
param containers array = []

resource stg 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    accessTier: 'Hot'
  }
}

resource blob 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  name: '${stg.name}/default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = [
  for c in containers: {
    name: '${stg.name}/default/${c}'
    properties: {
      publicAccess: 'None'
    }
    dependsOn: [
      blob
    ]
  }
]

output name string = stg.name
output id string = stg.id
output blobEndpoint string = stg.properties.primaryEndpoints.blob

