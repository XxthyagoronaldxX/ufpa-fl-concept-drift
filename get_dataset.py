import kagglehub

# Download latest version
path = kagglehub.dataset_download("mubashirrahim/wind-power-generation-data-forecasting", output_dir="./data")

print("Path to dataset files:", path)
