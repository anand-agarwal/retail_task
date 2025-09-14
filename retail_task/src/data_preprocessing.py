from bases.base import BaseModel
import pandas as pd
import numpy as np


class RetailModel(BaseModel):
    folder_name = 'retail_task'


    def preprocess(self):
        print('preprocessing')
        # First we drop all vars we are not considering:
        to_drop = ['transaction_id', 'transaction_date', 'transaction_hour',
                   'day_of_week', 'week_of_year', 'last_purchase_date', 'preferred_store',
                   'customer_zip_code', 'customer_city', 'store_zip_code', 'store_city',
                   'season']

        # Drop all columns starting with product
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('product')]
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('promotion')]
        self.df = self.df.drop(columns=to_drop)

        # One hot encode the columns
        one_hot = ['gender', 'income_bracket', 'marital_status', 'education_level', 'occupation', 'payment_method',
                   'store_location', 'purchase_frequency', 'customer_state', 'store_state',
                   'app_usage', 'social_media_engagement']

        double = ['email_subscriptions', 'weekend', 'holiday_season', 'churned', 'loyalty_program']

        for col in one_hot:
            self.one_hot_encode(col)

        for col in double:
            self.double_encode(col)
        #
        # to_keep = ["purchase_frequency", "avg_discount_used", "avg_items_per_transaction", "avg_transaction_value",
        #            "avg_purchase_value"]
        # self.df = self.df[to_keep]
        # self.df = self.df.dropna()
        # self.one_hot_encode("purchase_frequency")
        self.standardize(target="avg_purchase_value")
        print("standardization done")







