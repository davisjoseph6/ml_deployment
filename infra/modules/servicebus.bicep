@description('Service Bus namespace name (6-50).')
param name string

param location string
param tags object = {}

param sku string = 'Standard'

@description('Queues to create in the namespace.')
param queues array = []

resource sb 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
    tier: sku
    capacity: 1
  }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

resource sbQueues 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = [
  for q in queues: {
    name: '${sb.name}/${q}'
    properties: {
      enablePartitioning: true
      lockDuration: 'PT1M'
      defaultMessageTimeToLive: 'P14D'
    }
  }
]

output name string = sb.name
output id string = sb.id

