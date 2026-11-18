// API endpoints (Azure Container Apps). Edit here if the apps are redeployed.
window.FRAUD_API = {
  regions: [
    { id: "eastasia", name: "East Asia (Hong Kong)",
      url: "https://ca-fraud-eastasia.wittysand-fc563556.eastasia.azurecontainerapps.io" },
    { id: "centralindia", name: "Central India (Pune)",
      url: "https://ca-fraud-centralindia.nicemoss-c5d7190c.centralindia.azurecontainerapps.io" },
  ],
  samplesUrl: "samples.json",
  maxBatch: 1000,
};
