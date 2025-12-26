targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Environment name (dev|staging|prod).')
param env string = 'prod'

@description('Project prefix used in resource names.')
param prefix string = 'pcbqc'

@description('Optional tags applied to resources.')
param tags object = {
  project: prefix
  env: env
}

var uniq = toLower(uniqueString(resourceGroup().id, env))
var shortUniq = take(uniq, 6)

// Name constraints:
// - Storage: 3-24, lower+digits only
// - ACR: 5-50, alnum only
// - KV: 3-24, alnum+hyphen
// - SB: 6-50, alnum+hyphen
// - AKS: <=63, alnum+hyphen
var storageName = toLower(take('${prefix}${env}st${uniq}', 24))
var acrName = toLower(take('${prefix}${env}acr${uniq}', 50))
var kvName = toLower(take('${prefix}-${env}-kv-${shortUniq}', 24))
var sbName = toLower(take('${prefix}-${env}-sb-${shortUniq}', 50))
var aksName = toLower(take('${prefix}-${env}-aks-${shortUniq}', 63))

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    name: storageName
    location: location
    tags: tags
    containers: [
      'models'
      'data'
      'logs'
    ]
  }
}

module acr 'modules/acr.bicep' = {
  name: 'acr'
  params: {
    name: acrName
    location: location
    tags: tags
    sku: 'Standard'
  }
}

module keyvault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    name: kvName
    location: location
    tags: tags
    enableRbacAuthorization: true
  }
}

module servicebus 'modules/servicebus.bicep' = {
  name: 'servicebus'
  params: {
    name: sbName
    location: location
    tags: tags
    queues: [
      'jobs'
      'events'
    ]
  }
}

module aks 'modules/aks.bicep' = {
  name: 'aks'
  params: {
    name: aksName
    location: location
    tags: tags
    kubernetesVersion: '' // keep empty unless you verified a supported version in your region
    systemNodeCount: 1
    systemVmSize: 'Standard_D2s_v3'
    userNodeCount: 1
    userVmSize: 'Standard_D4s_v3'
  }
}

// Needed because roleAssignments.scope wants a resource symbol
resource acrExisting 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

// Give AKS kubelet identity AcrPull on the registry
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acrExisting.id, aksName, 'AcrPull')
  scope: acrExisting
  properties: {
    // AcrPull role definition id
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
    principalId: aks.outputs.kubeletObjectId
    principalType: 'ServicePrincipal'
  }
  dependsOn: [
    acr
  ]
}

output storageAccountName string = storage.outputs.name
output storageBlobEndpoint string = storage.outputs.blobEndpoint

output acrName string = acr.outputs.name
output acrLoginServer string = acr.outputs.loginServer

output keyVaultName string = keyvault.outputs.name
output keyVaultUri string = keyvault.outputs.vaultUri

output serviceBusNamespace string = servicebus.outputs.name

output aksClusterName string = aks.outputs.name
output aksKubeletObjectId string = aks.outputs.kubeletObjectId
