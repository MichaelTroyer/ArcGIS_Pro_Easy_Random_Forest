# -*- coding: utf-8 -*-

"""
 _____
|  ___|
| |__  __ _ ___ _   _
|  __|/ _` / __| | | |
| |__| (_| \__ \ |_| |
\____/\__,_|___/\__, |
                 __/ |
                |___/
______                _
| ___ \              | |
| |_/ /__ _ _ __   __| | ___  _ __ ___
|    // _` | '_ \ / _` |/ _ \| '_ ` _ \
| |\ \ (_| | | | | (_| | (_) | | | | | |
\_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|


______                  _
|  ___|                | |
| |_ ___  _ __ ___  ___| |_
|  _/ _ \| '__/ _ \/ __| __|
| || (_) | | |  __/\__ \ |_
\_| \___/|_|  \___||___/\__|


Even an archaeologist can do it!

Michael Troyer
michael.troyer@usda.gov
5/27/2020

Purpose: Model stuff..



"""


import os
import traceback
import pandas as pd
import arcpy

from datetime import datetime
from re import sub
from math import sqrt
from random import seed, randrange


#--- Helper Functions --------------------------------------------------------------------------------------------------

def get_acres(fc):
    """Check for an acres field in fc - create if doesn't exist or flag for calculation.
       Recalculate acres and return name of acre field"""

    # Add ACRES field to analysis area - check if exists
    field_list = [field.name for field in arcpy.ListFields(fc, wild_card='Acres')]

    # If *Acres* exists in table, flag for calculation instead
    if field_list:
        acre_field = field_list[0] # select the 'Acres' variant
    else:
        acre_field = "ACRES"
        arcpy.AddField_management(fc, acre_field, "DOUBLE", 15, 2)
    arcpy.CalculateField_management(fc, acre_field, "!shape.area@ACRES!", "PYTHON_9.3")
    acres_sum = sum([row[0] for row in arcpy.da.SearchCursor(fc, acre_field)])
    return {'Acres_Field': acre_field, 'Acres_Sum': acres_sum}


def build_where_clause(table, field, valueList):
    """Takes a list of values and constructs a SQL WHERE
    clause to select those values within a given field and table."""
    fieldDelimited = arcpy.AddFieldDelimiters(arcpy.Describe(table).path, field)
    fieldType = arcpy.ListFields(table, field)[0].type
    if str(fieldType) == 'String':
        valueList = ["'%s'" % value for value in valueList]
    whereClause = "%s IN(%s)" % (fieldDelimited, ', '.join(map(str, valueList)))
    return whereClause


def feature_class_to_data_frame(in_table, filter_fields=None):
    """
    Convert all or a subset of fields in an arcgis table into a pandas dataframe with an object ID index.
    """
    OIDFieldName = arcpy.Describe(in_table).OIDFieldName
    if filter_fields:
        final_fields = [OIDFieldName] + filter_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_table)]
    data = [row for row in arcpy.da.SearchCursor(in_table, final_fields)]
    fc_dataframe = pd.DataFrame(data, columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName, drop=True)
    return fc_dataframe


def one_hot_dataframe(dataframe, one_hot_columns):
    one_hot_dfs = []
    for col in one_hot_columns:
        one_hot_dfs.append(pd.get_dummies(dataframe[col], prefix=str(col)))
    dataframe.drop(columns=one_hot_columns, inplace=True)
    return dataframe.join(one_hot_dfs)


# Random Forest Functions ---------------------------------------------------------------------------------------------

# Create a random subsample from the dataset with replacement
def subsample(dataset, sample_proportion):
    sample = list()
    n_sample = round(len(dataset) * sample_proportion)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def get_best_split(dataset, n_features):
    # Last column is the class ID
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Check if a split group is all the same class
def is_homogeneous(group):
    return True if len({row[-1] for row in group}) == 1 else False


# Create a terminal node value
def to_terminal(group):
    # At a root node - return most frequent class
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Make a prediction with a decision tree
def predict(node, row):
    # less than = left
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            # Terminal nodes are not dicts, but values
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if is_homogeneous(left) or len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if is_homogeneous(right) or len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_best_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_proportion, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_proportion)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (trees, predictions)


def predict_dataset(trees, rows):
    return [bagging_predict(trees, row) for row in rows]


### Validation Functions ----------------------------------------------------------------------------------------------

# Split a dataset into k-folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def cross_validate(dataset, n_folds, max_depth, min_size, sample_proportion, n_trees, n_features):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        # Remove the test fold and flatten the training folds into a single list
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [row for fold in train_set for row in fold]
        test_set = list()
        # Wipe the class ID from the test data copy
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        trees, predicted = random_forest(
            train_set,
            test_set,
            max_depth,
            min_size,
            sample_proportion,
            n_trees,
            n_features
            )
        actual = [row[-1] for row in fold]
        assessment_metrics = confusion_matrix(actual, predicted)
        scores.append(assessment_metrics)
    return scores


def confusion_matrix(actual, predicted):
    total = len(actual)
    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predicted, name='Predicted')
    cm = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    tn = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tp = cm[1][1]

    accuracy = (tp+tn) / total                      # Overall, how often is the classifier correct?
    misclass_rate = (fp+fn) / total                 # Overall, how often is it wrong?
    tp_rate = tp / (tp+fn)                          # When it's actually yes, how often does it predict yes?  -  Sensitivity/Recall
    fp_rate = fp / (fp+tn)                          # When it's actually no, how often does it predict yes?
    tn_rate = tn / (tn+fp)                          # When it's actually no, how often does it predict no?  -  Specificity
    precision = tp / (tp+fp)                        # When it predicts yes, how often is it correct?
    prevalence = (fn+tp) / total                    # How often does the yes condition actually occur in the sample?
    null_error_rate = min(tp+fn, tn+fp) / total     # Error rate when guessing the majority class

    f_score = 2 * ((precision * tp_rate) / (precision + tp_rate))

    p_yes = ( ((tp+fn) / total) * ((tp+fp) / total) )
    p_no =  ( ((fp+tn) / total) * ((fn+tn) / total) )
    pe = p_yes + p_no
    cohens_kappa = 1 - ((1-accuracy) / (1-pe))

    return {
        'Confusion Matrix': cm,
        'True Negatives': tn,
        'True Positives': tp,
        'False Negatives': fn,
        'False Positives': fp,
        'Accuracy' : accuracy,
        'Misclassification Rate': misclass_rate,
        'True Positive Rate': tp_rate,
        'False Positive Rate': fp_rate,
        'True Negative Rate': tn_rate,
        'Precision': precision,
        'Prevalence': prevalence,
        'Null Error Rate': null_error_rate,
        'F-Score': f_score,
        'Cohens Kappa': cohens_kappa,
    }


#--- Custom Exceptions ------------------------------------------------------------------------------------------------

class InsufficientSurveyCoverage (BaseException): pass
class InsufficientSiteSample (BaseException): pass
class ExcessPoints (BaseException): pass
class GuessedTheMean (BaseException): pass


#--- Main Program -----------------------------------------------------------------------------------------------------

class Toolbox(object):
    def __init__(self):
        self.label = "Easy Random Forest Toolbox"
        self.alias = "Easy_Random_Forest_Toolbox"
        self.tools = [Easy_Random_Forest]


class Easy_Random_Forest(object):
    def __init__(self):
        self.label = "Easy Random Forest"
        self.description = ""
        self.canRunInBackground = True

    def getParameterInfo(self):

        # Input Analysis Area
        input_analysis_area=arcpy.Parameter(
            displayName="Input Modelling Analysis Area",
            name="Input_Analysis_Area",
            datatype="DEFeatureClass",
            )

        # Input Surveys feature class
        surveys=arcpy.Parameter(
            displayName="Existing Surveys Feature Class",
            name="Input_Surveys",
            datatype="DEFeatureClass",
            )

        # Surveys Subselection boolean
        surveys_filter=arcpy.Parameter(
            displayName="Filter Surveys",
            name="Filter_Surveys",
            datatype="Boolean",
            parameterType="Optional",
            enabled = "False",
            )

        # Surveys Subselection field
        surveys_filter_field=arcpy.Parameter(
            displayName="Surveys Filter Field",
            name="Surveys_Filter_Field",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Surveys Subselection value
        surveys_filter_value=arcpy.Parameter(
            displayName="Surveys Filter Value",
            name="Surveys_Filter_Value",
            datatype="String",
            parameterType="Optional",
            )

        # Input Sites feature class
        sites=arcpy.Parameter(
            displayName="Existing Sites Feature Class",
            name="Input_Sites",
            datatype="DEFeatureClass",
            )

        # Sites Subselection boolean
        sites_filter=arcpy.Parameter(
            displayName="Filter Sites",
            name="Filter_Sites",
            datatype="Boolean",
            parameterType="Optional",
            enabled = "False",
            )

        # Sites Subselection field
        sites_filter_field=arcpy.Parameter(
            displayName="Sites Filter Field",
            name="Sites_Filter_Field",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Sites Subselection value
        sites_filter_value=arcpy.Parameter(
            displayName="Sites Filter Value",
            name="Sites_Filter_Value",
            datatype="String",
            parameterType="Optional",
            enabled = "False",
            )

        # Input Analysis Feature Classes
        analysis_features=arcpy.Parameter(
            displayName="Analysis Feature Classes",
            name="Analysis_Feature_Classes",
            datatype="GPValueTable",
            parameterType='Optional',
            )
        analysis_features.columns = [['DEFeatureClass', 'Feature Class'], ['GPString', 'Field']]
        analysis_features.filters[1].type = 'ValueList'
        analysis_features.filters[1].list = ['Field']

        # Input Analysis Rasters
        analysis_rasters=arcpy.Parameter(
            displayName="Analysis Rasters",
            name="Analysis_Rasters",
            datatype="DERasterDataset",
            parameterType='Optional',
            multiValue=True,
            )

        # Sample point spacing
        sample_spacing=arcpy.Parameter(
            displayName="Sample Point Spacing (m)",
            name="Sample_Point_Spacing",
            datatype="Long",
            )

        # N-trees
        n_trees=arcpy.Parameter(
            displayName="Number of Trees",
            name="Number_of_Trees",
            datatype="Long",
            )

        # Max depth
        max_depth=arcpy.Parameter(
            displayName="Maximum Tree Depth",
            name="Maximum_Tree_Depth",
            datatype="Long",
            )

        max_features=arcpy.Parameter(
            displayName="Maximum Features",
            name="Maximum_Features",
            datatype="Long",
            )

        # Sample sizes
        sample_prop=arcpy.Parameter(
            displayName="Bootstrap Sample Proportion",
            name="Bootstrap_Sample_Proportion",
            datatype="Double",
            )

        # Min samples per split
        min_split=arcpy.Parameter(
            displayName="Minimum Samples per Split",
            name="Minimum_Samples_per_Split",
            datatype="Long",
            )

        # N-folds
        k_folds=arcpy.Parameter(
            displayName="Cross-Validation Folds",
            name="Cross_Validation_Folds",
            datatype="Long",
            )

        # Validation Test Proportion
        test_proportion=arcpy.Parameter(
            displayName="Final Validation Test Proportion",
            name="Test_Proportion",
            datatype="Double",
            )

        # Random seed
        random_seed=arcpy.Parameter(
            displayName="Random Seed",
            name="Random_Seed",
            datatype="Long",
            )

        # Output Location
        output_folder=arcpy.Parameter(
            displayName="Output Folder",
            name="Output_Folder",
            datatype="DEFolder",
            )

        return [
            input_analysis_area,
            surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
            sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
            sample_spacing, n_trees, max_depth, max_features, sample_prop, min_split, k_folds, test_proportion, random_seed,
            output_folder,
            ]


    def isLicensed(self):
        return True


    def updateParameters(self, params):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, max_features, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        input_analysis_area.filter.list = ["Polygon"]
        surveys.filter.list = ["Polygon"]
        sites.filter.list = ["Polygon"]

        # Subselect Surveys
        if surveys.value:
            surveys_filter.enabled = "True"
        else:
            surveys_filter.enabled = "False"

        if surveys_filter.value:
            surveys_fields = arcpy.Describe(surveys.value).fields
            surveys_field_list = [f.name for f in surveys_fields if f.type in ["String", "Integer", "SmallInteger"]]
            surveys_filter_field.enabled = "True"
            surveys_filter_field.filter.type = "ValueList"
            surveys_filter_field.filter.list = surveys_field_list
        else:
            surveys_filter_field.value = ""
            surveys_filter_field.enabled = "False"

        if surveys_filter_field.value:
            surveys_vals = set([r[0] for r in arcpy.da.SearchCursor(surveys.value, surveys_filter_field.value)])
            surveys_filter_value.enabled = True
            surveys_filter_value.filter.type = "ValueList"
            surveys_filter_value.filter.list = sorted([v for v in surveys_vals if v])
        else:
            surveys_filter_value.value = ""
            surveys_filter_value.enabled = False

        # Subselect Sites
        if sites.value:
            sites_filter.enabled = "True"
        else:
            sites_filter.enabled = "False"

        if sites_filter.value:
            sites_fields = arcpy.Describe(sites.value).fields
            sites_field_list = [f.name for f in sites_fields if f.type in ["String", "Integer", "SmallInteger"]]
            sites_filter_field.enabled = "True"
            sites_filter_field.filter.type = "ValueList"
            sites_filter_field.filter.list = sites_field_list
        else:
            sites_filter_field.value = ""
            sites_filter_field.enabled = "False"

        if sites_filter_field.value:
            sites_vals = set([r[0] for r in arcpy.da.SearchCursor(sites.value, sites_filter_field.value)])
            sites_filter_value.enabled = True
            sites_filter_value.filter.type = "ValueList"
            sites_filter_value.filter.list = sorted([v for v in sites_vals if v])
        else:
            sites_filter_value.value = ""
            sites_filter_value.enabled = False

        # Filter analysis input fields
        if analysis_features.value and analysis_features.altered:
            analysis_fields = []
            for fc, _ in analysis_features.values:
                analysis_fields.extend(
                    [f.name for f in arcpy.Describe(fc).fields if f.type in ["String", "Integer", "SmallInteger"]]
                    )
            analysis_fields = list(set(analysis_fields))
            analysis_fields.sort()
            analysis_features.filters[1].list = analysis_fields

        if not sample_spacing.altered: sample_spacing.value = 50
        if not n_trees.altered: n_trees.value = 5
        if not max_depth.altered: max_depth.value = 5
        if not max_features.altered: max_features.value = 5
        if not sample_prop.altered: sample_prop.value = 0.2
        if not min_split.altered: min_split.value = 10
        if not k_folds.altered: k_folds.value = 3
        if not test_proportion.altered: test_proportion.value = 0.2
        if not random_seed.altered: random_seed.value = 42


    def updateMessages(self, params):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, max_features, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        if analysis_features.altered:
            for fc, val in analysis_features.values:
                if val not in [f.name for f in arcpy.Describe(fc).fields]:
                    error_msg = ("Field [{}] not found in feature class [{}]".format(val, os.path.basename(str(fc))))
                    analysis_features.setErrorMessage(error_msg)

            fc_paths, fc_fields = zip(*[(str(path), str(field)) for path, field in analysis_features.values])
            # fc_paths = [str(path) for path, _ in analysis_features.values]
            # fc_fields = [str(field) for _, field in analysis_features.values]

            if len(set(fc_paths)) != len(fc_paths):
                err_mg = ("Duplicate input feature classes: the same feature class cannnot be used more than once.")
                analysis_features.setErrorMessage(err_mg)
            if len(set(fc_fields)) != len(fc_fields):
                err_mg = ("Duplicate input feature classes field names: the same feature class field name cannnot be used more than once.")
                analysis_features.setErrorMessage(err_mg)

        if analysis_rasters.value:
            raster_paths = [str(path) for path in analysis_rasters.valueAsText.split(';')]
            if len(set(raster_paths)) != len(raster_paths):
                err_mg = ("Duplicate input rasters: the same raster cannnot be used more than once.")
                analysis_rasters.setErrorMessage(err_mg)

        if sample_prop.value and sample_prop.value >=1:
            err_mg = ("Bootstrap Sample Proportion must be a number between 0 and 1.")
            sample_prop.setErrorMessage(err_mg)

        if test_proportion.value and test_proportion.value >=0.5:
            err_mg = ("Validation Test Proportion must be a number between 0 and 0.5")
            sample_prop.setErrorMessage(err_mg)

        # If output is supplied but no input analysis features were identifed.
        if output_folder.value and (not analysis_features.value and not analysis_rasters.value):
            err_mg = ("No input feature classes or rasters supplied for analysis.")
            analysis_features.setErrorMessage(err_mg)
            analysis_rasters.setErrorMessage(err_mg)

        return


    def execute(self, params, messages):
        (input_analysis_area,
        surveys, surveys_filter, surveys_filter_field, surveys_filter_value,
        sites, sites_filter, sites_filter_field, sites_filter_value, analysis_features, analysis_rasters,
        sample_spacing, n_trees, max_depth, max_features, sample_prop, min_split, k_folds, test_proportion, random_seed,
        output_folder,
        ) = params

        # Seed the RNG
        seed(random_seed.value)

        try:
            #--- Create Geodatabase ----------------------------------------------------------------------------------------------

            # Date/time stamp for outputs
            dt_stamp = sub('[^0-9]', '', str(datetime.now())[:16])

            # Output fGDB name and full path
            file_name = os.path.splitext(os.path.basename(__file__))[0] # Name outputs with tool name and date
            gdb_name = "{}_{}".format(file_name, dt_stamp)
            gdb_path = os.path.join(output_folder.valueAsText, gdb_name + '.gdb')

            # Create a geodatabase
            arcpy.CreateFileGDB_management(output_folder.valueAsText, gdb_name, "10.0")

            # Set workspace to fGDB
            arcpy.env.workspace = gdb_name

            # Get input analysis area spatial reference for outputs
            spatial_ref = arcpy.Describe(input_analysis_area.value).spatialReference

            #--- Get Anaysis Area -------------------------------------------------------------------------------------

            # Copy analysis area and get acres
            analysis_area = os.path.join(gdb_path, 'Analysis_Area')
            arcpy.CopyFeatures_management(input_analysis_area.value, analysis_area)
            analysis_acres_total = get_acres(analysis_area)['Acres_Sum']
            arcpy.AddMessage(f'Total acres within analysis area: {round(analysis_acres_total, 4)}')

            #--- Get Survey Data --------------------------------------------------------------------------------------

            # Check for a subselection - must supply all three values!
            if (surveys_filter.value and surveys_filter_field.value and surveys_filter_value.value):
                where = build_where_clause(surveys.value, surveys_filter_field.value, [surveys_filter_value.value])
                tmp = arcpy.management.MakeFeatureLayer(surveys.value, "memory\\surveys", where)
            else:
                tmp = arcpy.management.MakeFeatureLayer(surveys.value, "memory\\surveys")

            # Clip to analysis area
            arcpy.analysis.Clip("memory\\surveys", analysis_area, "memory\\clipped_surveys")

            # Dissolve and get survey acreage
            analysis_surveys = os.path.join(gdb_path, 'Analysis_Surveys')
            arcpy.Dissolve_management("memory\\clipped_surveys", analysis_surveys)
            survey_acres_total = get_acres(analysis_surveys)['Acres_Sum']

            survey_coverage = survey_acres_total / analysis_acres_total
            arcpy.AddMessage('Survey acres within analysis area: {}'.format(round(survey_acres_total, 2)))
            arcpy.AddMessage('Survey proportion within analysis area: {}\n'.format(round(survey_coverage, 3)))

            # Enforce minimum survey coverage for analysis
            if survey_coverage < 0.05:
                raise InsufficientSurveyCoverage

            arcpy.management.Delete("memory\\surveys")
            arcpy.management.Delete("memory\\clipped_surveys")

            #--- Get Site Data ----------------------------------------------------------------------------------------

            # Check for a subselection - must supply all three values!
            if (sites_filter.value and sites_filter_field.value and sites_filter_value.value):
                where = build_where_clause(sites.value, sites_filter_field.value, [sites_filter_value.value])
                tmp = arcpy.management.MakeFeatureLayer(sites.value, "memory\\sites", where)
            else:
                tmp = arcpy.management.MakeFeatureLayer(sites.value, "memory\\sites")

            # Clip to surveyed coverage
            analysis_sites =  os.path.join(gdb_path, 'Analysis_Sites')
            arcpy.analysis.Clip("memory\\sites", analysis_surveys, analysis_sites)

            site_count = int(arcpy.GetCount_management(analysis_sites).getOutput(0))
            site_density = round(site_count/survey_acres_total, 4)
            acres_per_site = round(1/site_density, 2)

            arcpy.AddMessage('Sites identified for analysis: {}'.format(site_count))
            arcpy.AddMessage('Site density in surveyed areas (sites/acre): {}'.format(site_density))
            arcpy.AddMessage('Approximately 1 site every {} acres'.format(acres_per_site))

            if site_count < 30:
                raise InsufficientSiteSample

            arcpy.management.Delete("memory\\sites")

            #--- Create Sample Datasets -------------------------------------------------------------------------------

            # Get the analysis area extent coordinates
            desc = arcpy.Describe(analysis_area)
            xmin, xmax = desc.extent.XMin, desc.extent.XMax
            ymin, ymax = desc.extent.YMin, desc.extent.YMax

            # Create fishnet
            arcpy.AddMessage('Creating sample point cloud')
            fishnet_poly = os.path.join(gdb_path, 'Fishnet')
            fishnet_points = os.path.join(gdb_path, 'Fishnet_label')
            sample_points = os.path.join(gdb_path, 'Sample_Points')
            arcpy.management.CreateFishnet(
                out_feature_class=fishnet_poly,
                origin_coord="{} {}".format(xmin, ymin),  # lower left
                y_axis_coord="{} {}".format(xmin, ymin+10),  # lower left + 10 meters to orient true north
                cell_width=sample_spacing.value,
                cell_height=sample_spacing.value,
                corner_coord="{} {}".format(xmax, ymax),  # upper right
                labels=True,
                template=analysis_area,
                geometry_type="POLYGON"
                )

            arcpy.analysis.Clip(fishnet_points, analysis_area, sample_points)
            arcpy.management.Delete(fishnet_poly)
            arcpy.management.Delete(fishnet_points)

            site_points = os.path.join(gdb_path, 'Site_Points')

            # Over sample the sites to ensure coverage - class balancing
            site_acres_total = get_acres(analysis_sites)['Acres_Sum']
            site_sample_points_per_acres = 10
            arcpy.management.CreateRandomPoints(
                out_path=gdb_path,
                out_name=os.path.basename(site_points),
                constraining_feature_class=analysis_sites,
                number_of_points_or_field=(site_acres_total * site_sample_points_per_acres) / site_count,
                minimum_allowed_distance="{} Meters".format(sqrt(sample_spacing.value)),
                )

            # Append site oversample to sample points and delete
            arcpy.management.Append(site_points, sample_points, 'NO_TEST')
            arcpy.management.Delete(site_points)

            n_points = int(arcpy.GetCount_management(sample_points).getOutput(0))
            arcpy.AddMessage(f'{n_points} sample points created')
            arcpy.AddMessage(f'Sample point density (points/acre): {round(n_points / analysis_acres_total, 2)}')

            if n_points > 250000:
                raise ExcessPoints

        #--- Data Attribution -----------------------------------------------------------------------------------------

            # Feature Classes ---------------------------------------------------------------------
            if analysis_features.value:

                arcpy.AddMessage('Gathering feature class attribute data')
                analysis_feature_classes, analysis_feature_fields = zip(*analysis_features.values)
                analysis_points = os.path.join(gdb_path, 'Analysis_Points')

                arcpy.Intersect_analysis(
                    in_features=analysis_feature_classes + (sample_points,),
                    out_feature_class=analysis_points,
                    )

                arcpy.management.Delete(sample_points)

                # clean up fields
                for field in arcpy.ListFields(analysis_points):
                    if field.name not in analysis_feature_fields + ('OBJECTID', 'Shape'):
                        arcpy.DeleteField_management(analysis_points, field.name)

            # Rasters -----------------------------------------------------------------------------
            if analysis_rasters.value:
                arcpy.AddMessage('Gathering raster attribute data')
                arcpy.sa.ExtractMultiValuesToPoints(analysis_points, analysis_rasters.valueAsText.split(';'))

        #--- Prep the dataframes --------------------------------------------------------------------------------------

            arcpy.AddMessage('Preparing the model')

            # Add class ID field
            class_field = 'Site_Point'
            arcpy.management.AddField(analysis_points, class_field, 'Short')

            # Add Surveyed / Prediction field
            predict_field = 'Predict'
            arcpy.management.AddField(analysis_points, predict_field, 'Short')

            analysis_points_lyr = arcpy.management.MakeFeatureLayer(analysis_points, 'memory\\analysis_points')

            # Encode site points
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_sites)
            site_points = int(arcpy.GetCount_management(analysis_points_lyr).getOutput(0))
            arcpy.management.CalculateField(analysis_points_lyr, class_field, '1')

            # Encode non-site points
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_sites, invert_spatial_relationship=True)
            non_site_points = int(arcpy.GetCount_management(analysis_points_lyr).getOutput(0))
            arcpy.management.CalculateField(analysis_points_lyr, class_field, '0')

            # Encode Surveyed points
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_surveys)
            arcpy.management.CalculateField(analysis_points_lyr, predict_field, '0')

            # Encode Prediction (non-surveyd) points
            arcpy.management.SelectLayerByLocation(analysis_points_lyr, 'INTERSECT', analysis_surveys, invert_spatial_relationship=True)
            arcpy.management.CalculateField(analysis_points_lyr, predict_field, '1')

            # Create the prediction point feature class
            prediction_points = os.path.join(gdb_path, 'Prediction_points')
            arcpy.management.CopyFeatures(analysis_points_lyr, prediction_points)
            for field in (class_field, predict_field):
                arcpy.DeleteField_management(prediction_points, field)

            # Prep the dataframe
            df = feature_class_to_data_frame(analysis_points)
            df.drop(columns=['Shape'], inplace=True)

            # Delete the prediction points and delete layer view
            arcpy.management.DeleteFeatures(analysis_points_lyr)
            arcpy.management.Delete(analysis_points_lyr)

            # Clean up analysis points fields
            arcpy.DeleteField_management(analysis_points, predict_field)

            # Try to coerce strings to numbers
            string_columns = df.select_dtypes(include=['object', 'category']).columns
            df[string_columns] = df[string_columns].apply(pd.to_numeric, errors='ignore', axis=1)

            # One hot remaining string columns
            one_hot_columns = df.select_dtypes(include=['object', 'category']).columns
            df = one_hot_dataframe(df, one_hot_columns)

            # Move the class_field data to the end of the DF where the RF will expect it!
            class_field_series = df[class_field]
            df.drop(columns=[class_field], inplace=True)
            df[class_field] = class_field_series

            # Remove the prediction set
            predict_df = df[df[predict_field] == 1]
            predict_df.drop(columns=[predict_field], inplace=True)
            df = df.drop(predict_df.index)
            df.drop(columns=[predict_field], inplace=True)

            # Create the Train / Test split - weighted to ensure representation of each class
            weights = df.groupby(class_field).count() * test_proportion.value
            neg_class_test_df = df[df[class_field] == 0].sample(int(weights.loc[0][0]), random_state=random_seed.value)
            pos_class_test_df = df[df[class_field] == 1].sample(int(weights.loc[1][0]), random_state=random_seed.value)
            test_df = pd.concat([neg_class_test_df, pos_class_test_df])
            train_df = df.drop(test_df.index)

            train_df.to_csv(os.path.join(output_folder.valueAsText, 'Train_df.csv'))
            test_df.to_csv(os.path.join(output_folder.valueAsText, 'Test_df.csv'))
            predict_df.to_csv(os.path.join(output_folder.valueAsText, 'Prediction_df.csv'))

            # Get train data as a list of lists
            train_dataset = train_df.values.tolist()

            # Remove class labels and get test data as a list of lists
            test_actuals = test_df[class_field].values.tolist()
            test_df.drop(columns=[class_field], inplace=True)
            test_dataset = test_df.values.tolist()

            # Max Features - must be less than number of features in df
            # TODO: Enforce don't use all? - SHould this even be an option?
            n_features = min(max_features.value, len(train_dataset[0]) - 1)
            # n_features = int(sqrt(len(train_dataset[0])-1))

            arcpy.AddMessage(f'Training model with {site_points} site points and {non_site_points} non-site points')
            site_class_proportion = (site_points / (site_points + non_site_points))
            site_class_percentage = round(site_class_proportion * 100, 2)
            non_site_class_percentage = round(100 - site_class_percentage, 2)
            arcpy.AddMessage(f'Class balance: {site_class_percentage}% site points, {non_site_class_percentage}% non-site points')

        # --- Random Forest --------------------------------------------------------------------------------------------

            if k_folds.value:
                # K-Fold cross-validation
                arcpy.AddMessage('Cross-validating model')
                cross_validation_scores = cross_validate(
                    train_dataset,
                    k_folds.value,
                    max_depth.value,
                    min_split.value,
                    sample_prop.value,
                    n_trees.value,
                    n_features
                    )
                arcpy.AddMessage(f'Cross-Validation Scores: {cross_validation_scores}')

            # Build the final model on the whole training set
            arcpy.AddMessage('Building the final model')
            final_model, final_predictions = random_forest(
                train_dataset,
                test_dataset,
                max_depth.value,
                min_split.value,
                sample_prop.value,
                n_trees.value,
                n_features,
                )
            # Final accuracy metric
            assessment_metrics = confusion_matrix(test_actuals, final_predictions)
            arcpy.AddMessage(assessment_metrics['Confusion Matrix'])
            for metric_name, val in sorted(assessment_metrics.items()):
                arcpy.AddMessage(f'{metric_name}: {val}')

            # TODO: Better way to handle?
            # Guard against no site predictions - throws list index error..
            if len(set(final_predictions)) == 1:
                raise GuessedTheMean

            # Predict the unsurveyed areas
            arcpy.AddMessage('Adding predictions to unsurveyed areas')
            predict_df.drop(columns=[class_field], inplace=True)
            prediction_dataset = predict_df.values.tolist()
            predictions = predict_dataset(final_model, prediction_dataset)

            arcpy.AddMessage(
                f'Unsurveyed area predictions: {predictions.count(1)} sites points, {predictions.count(0)} non-site points'
                )

            # Join predictions to original point data
            final_predict_field = 'Final_Prediction'
            arcpy.management.AddField(prediction_points, final_predict_field, 'Short')
            with arcpy.da.UpdateCursor(prediction_points, final_predict_field) as cur:
                for ix, row in enumerate(cur):
                    row[0] = predictions[ix]
                    cur.updateRow(row)


        #--- Error Handling -------------------------------------------------------------------------------------------

        except InsufficientSurveyCoverage:
            msg = (
                "Insufficient survey coverage in analysis area.\n"
                "Model requires a minimum of 5 percent survey coverage for analysis.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        except InsufficientSiteSample:
            msg = (
                "Too few sites in analysis area.\n"
                "Model requires a minimum of 30 sites for analysis.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        except ExcessPoints:
            msg = (
                "Too many points requested for analaysis.\n"
                "Adjust your sample spacing and try again.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        except GuessedTheMean:
            msg = (
                "Final model simply guessed the mean.\n"
                "Your feature class and raster inputs do not appear to have much predictive potential.\n"
                "Try adding new prediction feature classes or rasters and/or adjusting your sample spacing.\n"
                "[exiting program]")
            arcpy.AddError(msg)

        # except:
        #     err = str(traceback.format_exc())s
        #     arcpy.AddError(err)

        finally:
            arcpy.management.Delete('memory')

        return
