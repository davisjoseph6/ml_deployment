@description('AKS cluster name.')
param name string

param location string
param tags object = {}

@description('If empty, AKS uses default supported version.')
param kubernetesVersion string = ''

param systemNodeCount int = 1
param systemVmSize string = 'Standard_D2s_v3'

param userNodeCount int = 1
param userVmSize string = 'Standard_D4s_v3'

var maybeK8sVersion = empty(kubernetesVersion) ? {} : { kubernetesVersion: kubernetesVersion }

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
    agentPoolProfiles: [
      {
        name: 'sys'
        mode: 'System'
        count: systemNodeCount
        vmSize: systemVmSize
        osType: 'Linux'
        type: 'VirtualMachineScaleSets'
      }
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
      }
    ]
    networkProfile: {
      networkPlugin: 'kubenet'
      loadBalancerSku: 'standard'
      outboundType: 'loadBalancer'
    }
  })
}

output name string = aks.name
output id string = aks.id

// Kubelet identity (used for ACR pulls)
output kubeletObjectId string = aks.properties.identityProfile.kubeletidentity.objectId
output kubeletClientId string = aks.properties.identityProfile.kubeletidentity.clientId

