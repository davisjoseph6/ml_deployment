@description('ACR name (5-50 alphanumeric). Must be globally unique.')
param name string

param location string
param tags object = {}
param sku string = 'Standard'

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

output name string = acr.name
output id string = acr.id
output loginServer string = acr.properties.loginServer

