# create some polars data
import polars
# data = [(1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
#         (2.0, None, 2.0, 7.0, 8.5, 6.7), 
#         (2.0, None, 3.0, 7.0, -2.3, 4.4),
#         (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
#         (4.0, 5.0, 4.0, 7.0, 12.1, -5.2)]

# data = [([1.0, 1.0, 1.0], [1.0, 1.1, 1.2], [1.0, 1.1, 0.9], [7.0, 6.9, 7.2], [1.5], [-2.3, -2.3, -2.3]), 
#         ([2.0, 2.0, 2.0], [None, None, None], [2.0, 1.9, 1.8], [7.0, 7.1, 6.9], [8.5], [6.7, 6.8, 6.5]), 
#         ([2.0, 2.0, 2.0], [None, None, None], [3.0, 2.9, 2.8], [6.9, 7.0, 7.1], [-2.3], [4.4, 4.3, 4.2]),
#         ([3.0, 3.0, 3.0], [4.0, 3.9, 4.1], [3.0, 3.1, 2.9], [6.9, 7.0, 7.1], [0.0], [0.0, 0.1, 0.2]),
#         ([4.0, 4.0, 4.0], [5.0, 4.9, 5.1], [4.0, 3.9, 3.8], [7.0, 7.1, 6.9], [12.1], [-5.2, -5.1, -5.0])]

data = [((1.0, 1.5), (1.0, 1.1, 1.2), [1.0, 1.1, 0.9], [7.0, 6.9, 7.2], (1.5), [-2.3, -2.3, -2.3]), 
        ((2.0, 1.5), (None, None, None), [2.0, 1.9, 1.8], [7.0, 7.1, 6.9], (8.5), [6.7, 6.8, 6.5]), 
        ((2.0, 1.5), (None, None, None), [3.0, 2.9, 2.8], [6.9, 7.0, 7.1], (-2.3), [4.4, 4.3, 4.2]),
        ((3.0, 1.5), (4.0, 3.9, 4.1), [3.0, 3.1, 2.9], [6.9, 7.0, 7.1], (0.0), [0.0, 0.1, 0.2]),
        ((4.0, 1.5), (5.0, 4.9, 5.1), [4.0, 3.9, 3.8], [7.0, 7.1, 6.9], (12.1), [-5.2, -5.1, -5.0])]


columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]
df_polars = polars.DataFrame(data=data, schema=columns)

# select top 2 features using mRMR
import mrmr
selected_features = mrmr.polars.mrmr_regression(df=df_polars, target_column="target", K=2, return_scores=True)
print('\nselected_features: ', selected_features)