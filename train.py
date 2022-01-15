

models = {}
models['Linear'] = {'GSOs': 1}
models['SimpleGNN'] = {'GSOs': 1}
models['SimpleILGNN'] = {'GSOs': 1}
models['MultiChannelGNN'] = {'GSOs': 2}
models['MultiChannelILGNN'] = {'GSOs': 2}
models['MultigraphNN'] = {'GSOs': 2}
#models['MultigraphILNN'] = {'GSOs': 2}

datasets = []
for i in range(5):
  print(f'Creating dataset {str(i+1)}...')
  datasets.append(create_data(trainMovie = movie, minRatings = minRatings, trainProp = 0.8, validProp = 0.1, knn = 200, seed = i))