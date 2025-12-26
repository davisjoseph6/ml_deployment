@description('AKS cluster name.')
param name string

param location string
param tags object = {}

@description('If empty, AKS uses default supported version.')
param kubernetesVersion string = ''

param systemNodeCount int = 1
param systemVmSize string

param userNodeCount int = 0
param userVmSize string

var maybeK8sVersion = empty(kubernetesVersion) ? {} : { kubernetesVersion: kubernetesVersion }

var systemPool = [
  {
    name: 'sys'
    mode: 'System'
    count: systemNodeCount
    vmSize: systemVmSize
    osType: 'Linux'
    type: 'VirtualMachineScaleSets'
    upgradeSettings: {
      maxSurge: '0' // <-- was '1'
      maxUnavailable: '1'
    }
  }
]

var userPool = userNodeCount > 0 ? [
  {
    name: 'usr'
    mode: 'User'
    count: userNodeCount
    vmSize: userVmSize
    osType: 'Linux'
    type: 'VirtualMachineScaleSets'
    nodeLabels: {
      workload: 'user'
    }
    upgradeSettings: {
      maxSurge: '0'
      maxUnavailable: '1'
    }
  }
] : []

resource aks 'Microsoft.ContainerService/managedClusters@2023-05-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: union(maybeK8sVersion, {
    dnsPrefix: name
    enableRBAC: true
    agentPoolProfiles: concat(systemPool, userPool)
    networkProfile: {
      networkPlugin: 'kubenet'
      loadBalancerSku: 'standard'
      outboundType: 'loadBalancer'
    }
  })
}

output name string = aks.name
output id string = aks.id
output kubeletObjectId string = aks.properties.identityProfile.kubeletidentity.objectId
output kubeletClientId string = aks.properties.identityProfile.kubeletidentity.clientId

