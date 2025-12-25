@description('Name of the AKS cluster')
param aksName string

@description('Location of the resources')
param location string

@description('Node pool name for CPU nodes')
param cpuNodePoolName string = 'cpu'

@description('Node count for CPU nodes')
param cpuNodeCount int = 1

@description('Kubernetes version')
param kubernetesVersion string = '1.31.4'

@description('Resource group name')
param rgName string

resource aks 'Microsoft.ContainerService/managedClusters@2023-05-01' = {
  name: aksName
  location: location
  properties: {
    kubernetesVersion: kubernetesVersion
    dnsPrefix: aksName
    agentPoolProfiles: [
      {
        name: cpuNodePoolName
        count: cpuNodeCount
        vmSize: 'Standard_D2s_v3'
        mode: 'System'
        osType: 'Linux'
      }
    ]
    enableRBAC: true
  }
}
