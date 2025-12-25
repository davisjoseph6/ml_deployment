module aksModule 'modules/aks.bicep' = {
  name: 'aksDeployment'
  params: {
    aksName: 'aks-pcbqc-dev-neu-001'
    location: 'northeurope'
    rgName: 'rg-pcbqc-dev-neu-001'
  }
}
