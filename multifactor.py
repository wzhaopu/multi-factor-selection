from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
# Set to INFO for tracking training, default is WARN

print("Using TensorFlow version %s" % (tf.__version__))

CATEGORICAL_COLUMNS = ['exchangeCD','ListSectorCD']

# Columns of the input csv file
COLUMNS = ['ticker', 'dailyReturnNoReinv', 'exchangeCD','ListSectorCD' 'totalShares', 'nonrestFloatShares', 'nonrestfloatA', 'TShEquity',
            'AccountsPayablesTDays', 'AccountsPayablesTRate', 'AdminiExpenseRate', 'ARTDays', 'ARTRate', 'ASSI', 'BLEV',
           'BondsPayableToAsset', 'CashRateOfSales', 'CashToCurrentLiability', 'CMRA', 'CTOP', 'CTP5', 'CurrentAssetsRatio',
           'CurrentAssetsTRate', 'CurrentRatio', 'DAVOL10', 'DAVOL20', 'DAVOL5', 'DDNBT', 'DDNCR', 'DDNSR', 'DebtEquityRatio',
           'DebtsAssetRatio', 'DHILO', 'DilutedEPS', 'DVRAT', 'EBITToTOR', 'EGRO', 'EMA10', 'EMA120', 'EMA20', 'EMA5', 'EMA60',
           'EPS', 'EquityFixedAssetRatio', 'EquityToAsset', 'EquityTRate', 'ETOP', 'ETP5', 'FinancialExpenseRate', 'FinancingCashGrowRate',
           'FixAssetRatio', 'FixedAssetsTRate', 'GrossIncomeRatio', 'HBETA', 'HSIGMA', 'IntangibleAssetRatio', 'InventoryTDays',
           'InventoryTRate', 'InvestCashGrowRate', 'LCAP', 'LFLO', 'LongDebtToAsset', 'LongDebtToWorkingCapital', 'LongTermDebtToAsset',
           'MA10', 'MA120', 'MA20', 'MA5', 'MA60', 'MAWVAD', 'MFI', 'MLEV', 'NetAssetGrowRate', 'NetProfitGrowRate', 'NetProfitRatio',
           'NOCFToOperatingNI', 'NonCurrentAssetsRatio', 'NPParentCompanyGrowRate', 'NPToTOR', 'OperatingExpenseRate',
           'OperatingProfitGrowRate', 'OperatingProfitRatio', 'OperatingProfitToTOR', 'OperatingRevenueGrowRate', 'OperCashGrowRate',
           'OperCashInToCurrentLiability', 'PB', 'PCF', 'PE', 'PS', 'PSY', 'QuickRatio', 'REVS10', 'REVS20', 'REVS5', 'ROA', 'ROA5', 'ROE',
           'ROE5', 'RSI', 'RSTR12', 'RSTR24', 'SalesCostRatio', 'SaleServiceCashToOR', 'SUE', 'TaxRatio', 'TOBT', 'TotalAssetGrowRate',
           'TotalAssetsTRate', 'TotalProfitCostRatio', 'TotalProfitGrowRate', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 'VOL5', 'VOL60', 'WVAD',
           'TA2EV', 'CFO2EV', 'ACCA', 'DEGM', 'SUOI', 'EARNMOM', 'FiftyTwoWeekHigh', 'Volatility', 'Skewness', 'ILLIQUIDITY', 'BackwardADJ',
           'MACD', 'ADTM', 'ATR14', 'ATR6', 'BIAS10', 'BIAS20', 'BIAS5', 'BIAS60', 'BollDown', 'BollUp', 'CCI10', 'CCI20', 'CCI5', 'CCI88',
           'KDJ_K', 'KDJ_D', 'KDJ_J', 'ROC6', 'ROC20', 'SBM', 'STM', 'UpRVI', 'DownRVI', 'RVI', 'SRMI', 'ChandeSD', 'ChandeSU', 'CMO', 'DBCD',
           'ARC', 'OBV', 'OBV6', 'OBV20', 'TVMA20', 'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF', 'VEMA10', 'VEMA12', 'VEMA26', 'VEMA5',
           'VMACD', 'VOSC', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'KlingerOscillator', 'MoneyFlow20', 'AD', 'AD20', 'AD6',
           'CoppockCurve', 'ASI', 'ChaikinOscillator', 'ChaikinVolatility', 'EMV14', 'EMV6', 'plusDI', 'minusDI', 'ADX', 'ADXR', 'Aroon',
           'AroonDown', 'AroonUp', 'DEA', 'DIFF', 'DDI', 'DIZ', 'DIF', 'MTM', 'MTMMA', 'PVT', 'PVT6', 'PVT12', 'TRIX5', 'TRIX10', 'UOS',
           'MA10RegressCoeff12', 'MA10RegressCoeff6', 'PLRC6', 'PLRC12', 'SwingIndex', 'Ulcer10', 'Ulcer5', 'Hurst', 'ACD6', 'ACD20', 'EMA12',
           'EMA26', 'APBMA', 'BBI', 'BBIC', 'TEMA10', 'TEMA5', 'MA10Close', 'AR', 'BR', 'ARBR', 'CR20', 'MassIndex', 'BearPower', 'BullPower',
           'Elder', 'NVI', 'PVI', 'RC12', 'RC24', 'JDQS20']

# Feature columns for input into the model
FEATURE_COLUMNS = ['exchangeCD', 'ListSectorCD','totalShares', 'nonrestFloatShares', 'nonrestfloatA', 'TShEquity',
            'AccountsPayablesTDays', 'AccountsPayablesTRate', 'AdminiExpenseRate', 'ARTDays', 'ARTRate', 'ASSI', 'BLEV',
           'BondsPayableToAsset', 'CashRateOfSales', 'CashToCurrentLiability', 'CMRA', 'CTOP', 'CTP5', 'CurrentAssetsRatio',
           'CurrentAssetsTRate', 'CurrentRatio', 'DAVOL10', 'DAVOL20', 'DAVOL5', 'DDNBT', 'DDNCR', 'DDNSR', 'DebtEquityRatio',
           'DebtsAssetRatio', 'DHILO', 'DilutedEPS', 'DVRAT', 'EBITToTOR', 'EGRO', 'EMA10', 'EMA120', 'EMA20', 'EMA5', 'EMA60',
           'EPS', 'EquityFixedAssetRatio', 'EquityToAsset', 'EquityTRate', 'ETOP', 'ETP5', 'FinancialExpenseRate', 'FinancingCashGrowRate',
           'FixAssetRatio', 'FixedAssetsTRate', 'GrossIncomeRatio', 'HBETA', 'HSIGMA', 'IntangibleAssetRatio', 'InventoryTDays',
           'InventoryTRate', 'InvestCashGrowRate', 'LCAP', 'LFLO', 'LongDebtToAsset', 'LongDebtToWorkingCapital', 'LongTermDebtToAsset',
           'MA10', 'MA120', 'MA20', 'MA5', 'MA60', 'MAWVAD', 'MFI', 'MLEV', 'NetAssetGrowRate', 'NetProfitGrowRate', 'NetProfitRatio',
           'NOCFToOperatingNI', 'NonCurrentAssetsRatio', 'NPParentCompanyGrowRate', 'NPToTOR', 'OperatingExpenseRate',
           'OperatingProfitGrowRate', 'OperatingProfitRatio', 'OperatingProfitToTOR', 'OperatingRevenueGrowRate', 'OperCashGrowRate',
           'OperCashInToCurrentLiability', 'PB', 'PCF', 'PE', 'PS', 'PSY', 'QuickRatio', 'REVS10', 'REVS20', 'REVS5', 'ROA', 'ROA5', 'ROE',
           'ROE5', 'RSI', 'RSTR12', 'RSTR24', 'SalesCostRatio', 'SaleServiceCashToOR', 'SUE', 'TaxRatio', 'TOBT', 'TotalAssetGrowRate',
           'TotalAssetsTRate', 'TotalProfitCostRatio', 'TotalProfitGrowRate', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 'VOL5', 'VOL60', 'WVAD',
           'TA2EV', 'CFO2EV', 'ACCA', 'DEGM', 'SUOI', 'EARNMOM', 'FiftyTwoWeekHigh', 'Volatility', 'Skewness', 'ILLIQUIDITY', 'BackwardADJ',
           'MACD', 'ADTM', 'ATR14', 'ATR6', 'BIAS10', 'BIAS20', 'BIAS5', 'BIAS60', 'BollDown', 'BollUp', 'CCI10', 'CCI20', 'CCI5', 'CCI88',
           'KDJ_K', 'KDJ_D', 'KDJ_J', 'ROC6', 'ROC20', 'SBM', 'STM', 'UpRVI', 'DownRVI', 'RVI', 'SRMI', 'ChandeSD', 'ChandeSU', 'CMO', 'DBCD',
           'ARC', 'OBV', 'OBV6', 'OBV20', 'TVMA20', 'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF', 'VEMA10', 'VEMA12', 'VEMA26', 'VEMA5',
           'VMACD', 'VOSC', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'KlingerOscillator', 'MoneyFlow20', 'AD', 'AD20', 'AD6',
           'CoppockCurve', 'ASI', 'ChaikinOscillator', 'ChaikinVolatility', 'EMV14', 'EMV6', 'plusDI', 'minusDI', 'ADX', 'ADXR', 'Aroon',
           'AroonDown', 'AroonUp', 'DEA', 'DIFF', 'DDI', 'DIZ', 'DIF', 'MTM', 'MTMMA', 'PVT', 'PVT6', 'PVT12', 'TRIX5', 'TRIX10', 'UOS',
           'MA10RegressCoeff12', 'MA10RegressCoeff6', 'PLRC6', 'PLRC12', 'SwingIndex', 'Ulcer10', 'Ulcer5', 'Hurst', 'ACD6', 'ACD20', 'EMA12',
           'EMA26', 'APBMA', 'BBI', 'BBIC', 'TEMA10', 'TEMA5', 'MA10Close', 'AR', 'BR', 'ARBR', 'CR20', 'MassIndex', 'BearPower', 'BullPower',
           'Elder', 'NVI', 'PVI', 'RC12', 'RC24', 'JDQS20']

NUMERICAL_COLUMNS = ['totalShares', 'nonrestFloatShares', 'nonrestfloatA', 'TShEquity',
            'AccountsPayablesTDays', 'AccountsPayablesTRate', 'AdminiExpenseRate', 'ARTDays', 'ARTRate', 'ASSI', 'BLEV',
           'BondsPayableToAsset', 'CashRateOfSales', 'CashToCurrentLiability', 'CMRA', 'CTOP', 'CTP5', 'CurrentAssetsRatio',
           'CurrentAssetsTRate', 'CurrentRatio', 'DAVOL10', 'DAVOL20', 'DAVOL5', 'DDNBT', 'DDNCR', 'DDNSR', 'DebtEquityRatio',
           'DebtsAssetRatio', 'DHILO', 'DilutedEPS', 'DVRAT', 'EBITToTOR', 'EGRO', 'EMA10', 'EMA120', 'EMA20', 'EMA5', 'EMA60',
           'EPS', 'EquityFixedAssetRatio', 'EquityToAsset', 'EquityTRate', 'ETOP', 'ETP5', 'FinancialExpenseRate', 'FinancingCashGrowRate',
           'FixAssetRatio', 'FixedAssetsTRate', 'GrossIncomeRatio', 'HBETA', 'HSIGMA', 'IntangibleAssetRatio', 'InventoryTDays',
           'InventoryTRate', 'InvestCashGrowRate', 'LCAP', 'LFLO', 'LongDebtToAsset', 'LongDebtToWorkingCapital', 'LongTermDebtToAsset',
           'MA10', 'MA120', 'MA20', 'MA5', 'MA60', 'MAWVAD', 'MFI', 'MLEV', 'NetAssetGrowRate', 'NetProfitGrowRate', 'NetProfitRatio',
           'NOCFToOperatingNI', 'NonCurrentAssetsRatio', 'NPParentCompanyGrowRate', 'NPToTOR', 'OperatingExpenseRate',
           'OperatingProfitGrowRate', 'OperatingProfitRatio', 'OperatingProfitToTOR', 'OperatingRevenueGrowRate', 'OperCashGrowRate',
           'OperCashInToCurrentLiability', 'PB', 'PCF', 'PE', 'PS', 'PSY', 'QuickRatio', 'REVS10', 'REVS20', 'REVS5', 'ROA', 'ROA5', 'ROE',
           'ROE5', 'RSI', 'RSTR12', 'RSTR24', 'SalesCostRatio', 'SaleServiceCashToOR', 'SUE', 'TaxRatio', 'TOBT', 'TotalAssetGrowRate',
           'TotalAssetsTRate', 'TotalProfitCostRatio', 'TotalProfitGrowRate', 'VOL10', 'VOL120', 'VOL20', 'VOL240', 'VOL5', 'VOL60', 'WVAD',
           'TA2EV', 'CFO2EV', 'ACCA', 'DEGM', 'SUOI', 'EARNMOM', 'FiftyTwoWeekHigh', 'Volatility', 'Skewness', 'ILLIQUIDITY', 'BackwardADJ',
           'MACD', 'ADTM', 'ATR14', 'ATR6', 'BIAS10', 'BIAS20', 'BIAS5', 'BIAS60', 'BollDown', 'BollUp', 'CCI10', 'CCI20', 'CCI5', 'CCI88',
           'KDJ_K', 'KDJ_D', 'KDJ_J', 'ROC6', 'ROC20', 'SBM', 'STM', 'UpRVI', 'DownRVI', 'RVI', 'SRMI', 'ChandeSD', 'ChandeSU', 'CMO', 'DBCD',
           'ARC', 'OBV', 'OBV6', 'OBV20', 'TVMA20', 'TVMA6', 'TVSTD20', 'TVSTD6', 'VDEA', 'VDIFF', 'VEMA10', 'VEMA12', 'VEMA26', 'VEMA5',
           'VMACD', 'VOSC', 'VR', 'VROC12', 'VROC6', 'VSTD10', 'VSTD20', 'KlingerOscillator', 'MoneyFlow20', 'AD', 'AD20', 'AD6',
           'CoppockCurve', 'ASI', 'ChaikinOscillator', 'ChaikinVolatility', 'EMV14', 'EMV6', 'plusDI', 'minusDI', 'ADX', 'ADXR', 'Aroon',
           'AroonDown', 'AroonUp', 'DEA', 'DIFF', 'DDI', 'DIZ', 'DIF', 'MTM', 'MTMMA', 'PVT', 'PVT6', 'PVT12', 'TRIX5', 'TRIX10', 'UOS',
           'MA10RegressCoeff12', 'MA10RegressCoeff6', 'PLRC6', 'PLRC12', 'SwingIndex', 'Ulcer10', 'Ulcer5', 'Hurst', 'ACD6', 'ACD20', 'EMA12',
           'EMA26', 'APBMA', 'BBI', 'BBIC', 'TEMA10', 'TEMA5', 'MA10Close', 'AR', 'BR', 'ARBR', 'CR20', 'MassIndex', 'BearPower', 'BullPower',
           'Elder', 'NVI', 'PVI', 'RC12', 'RC24', 'JDQS20']


df = pd.read_csv("stock_factors.csv", header=None, names=COLUMNS)


BATCH_SIZE = 40






def generate_input_fn(filename, batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # Reads out batch_size number of lines
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)

        # record_defaults should match the datatypes of each respective column.
        record_defaults = [[" "], [0.0], [" "], [" "]]
        for i in range(0,237):
            record_defaults.append([0.0])
        #print(record_defaults)


        # Decode CSV data that was just read out.
        columns = tf.decode_csv(
            value, record_defaults=record_defaults)

        # features is a dictionary that maps from column names to tensors of the data.
        # income_bracket is the last column of the data. Note that this is NOT a dict.
        all_columns = dict(zip(COLUMNS, columns))

        # Save the income_bracket column as our labels
        # dict.pop() returns the popped array of income_bracket values
        dailyReturn = all_columns.pop('dailyReturnNoReinv')

        # remove the fnlwgt key, which is not used
        all_columns.pop('ticker', 'ticker key not found')

        # the remaining columns are our features
        features = all_columns


        # Sparse categorical features must be represented with an additional dimension.
        # There is no additional work needed for the Continuous columns; they are the unaltered columns.
        # See docs for tf.SparseTensor for more info
        for feature_name in CATEGORICAL_COLUMNS:
            # Requires tensorflow >= 0.12
            features[feature_name] = tf.expand_dims(features[feature_name], -1)

        # Convert ">50K" to 1, and "<=50K" to 0
        labels = tf.greater(dailyReturn, 0.0)

        return features, labels

    return _input_fn

print('input function configured')

# Sparse base columns.

exchangeCD = tf.contrib.layers.sparse_column_with_keys(column_name="exchangeCD",
                                                 keys=["XSHE", "XSHG"])
listSectorCD = tf.contrib.layers.sparse_column_with_keys(column_name="listSectorCD",
                                                 keys=["1","2","3","4"])

print('Sparse columns configured')

# Continuous base columns.

numerical_dict = {}
for col in NUMERICAL_COLUMNS:
    numerical_dict.update({col : tf.contrib.layers.real_valued_column(col)})

print('continuous columns configured')

'''
# Transformations.
education_occupation = tf.contrib.layers.crossed_column([education, occupation],
                                                        hash_bucket_size=int(1e4))
age_race_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation],
                                                       hash_bucket_size=int(1e6))
country_occupation = tf.contrib.layers.crossed_column([native_country, occupation],
                                                      hash_bucket_size=int(1e4))
'''
print('TODO: Transformations incomplete')

# Wide columns and deep columns.
wide_columns = [exchangeCD, listSectorCD]
deep_columns = [listSectorCD]

for col in NUMERICAL_COLUMNS:
    wide_columns.append(numerical_dict[col])
    deep_columns.append(tf.layers.embedding_column(numerical_dict[col]))

print('wide and deep columns configured')


def create_model_dir(model_type):
    return 'models/model_' + model_type + '_' + str(int(time.time()))


# If new_model=False, pass in the desired model_dir
def get_model(model_type, new_model=True, model_dir=None):
    if new_model or model_dir is None:
        model_dir = create_model_dir(model_type)  # Comment out this line to continue training a existing model
    print("Model directory = %s" % model_dir)

    m = None

    # Linear Classifier
    if model_type == 'WIDE':
        m = tf.contrib.learn.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)

    # Deep Neural Net Classifier
    if model_type == 'DEEP':
        m = tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])

    # Combined Linear and Deep Classifier
    if model_type == 'WIDE_AND_DEEP':
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25])

    print('estimator built')

    return m, model_dir


m, model_dir = get_model(model_type='WIDE_AND_DEEP')

train_file = "adult.data.csv" # "gs://cloudml-public/census/data/adult.data.csv"
test_file  = "adult.test.csv" # "gs://cloudml-public/census/data/adult.test.csv"


train_steps = 1000

m.fit(input_fn=generate_input_fn(train_file, BATCH_SIZE), steps=train_steps)

print('fit done')

esults = m.evaluate(input_fn=generate_input_fn(test_file), steps=100)
print('evaluate done')

print('Accuracy: %s' % results['accuracy'])