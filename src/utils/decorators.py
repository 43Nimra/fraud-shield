import sys
sys.path.insert(0, '.')

import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator — function kitna time leta hai measure karta hai.
    WHY: ML pipeline mein bottleneck find karna zaroori hai.
    Feature engineering slow hai ya model training?
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()
        elapsed = (end - start) * 1000

        logger.info(f"[TIMER] {func.__name__} → {elapsed:.2f}ms")

        return result

    return wrapper


def validate_dataframe(required_columns: list):
    """
    Decorator factory — DataFrame ke required columns validate karta hai.

    Supports both:

        Normal function:
            process_data(df)

        Class method:
            engineer.fit(df)
    """
    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # DataFrame keyword argument se pass hua ho
            df = kwargs.get('df')

            # Positional arguments mein DataFrame find karo.
            # Normal function:
            #     args = (df,)
            #
            # Class method:
            #     args = (self, df)
            if df is None:
                for arg in args:
                    if hasattr(arg, 'columns'):
                        df = arg
                        break

            if df is None:
                raise ValueError(
                    f"{func.__name__} ko DataFrame argument nahi mila"
                )

            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"{func.__name__} ke liye ye columns chahiye: {missing}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_step(func: Callable) -> Callable:
    """
    Decorator — pipeline step start/end log karta hai.
    WHY: Jab feature engineering fail ho — kahan fail hua pata chale.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        logger.info(f"[START] {func.__name__}")

        try:
            result = func(*args, **kwargs)

            logger.info(f"[DONE]  {func.__name__}")

            return result

        except Exception as e:
            logger.error(f"[FAIL]  {func.__name__} → {e}")
            raise

    return wrapper


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test 1 — @timer
    @timer
    def slow_function():
        time.sleep(0.1)
        return "done"

    result = slow_function()
    print(f"Result: {result}")

    # Test 2 — @validate_dataframe
    import pandas as pd

    @validate_dataframe(['amount', 'is_fraud'])
    @timer
    def process_data(df: pd.DataFrame) -> pd.DataFrame:
        return df

    # Valid df
    df_good = pd.DataFrame({
        'amount': [100, 200],
        'is_fraud': [0, 1]
    })

    process_data(df_good)
    print("Valid DataFrame: OK")

    # Invalid df — missing column
    try:
        df_bad = pd.DataFrame({
            'amount': [100]
        })

        process_data(df_bad)

    except ValueError as e:
        print(f"Caught: {e}")

    # Test 3 — @log_step
    @log_step
    @timer
    def feature_engineering_step():
        time.sleep(0.05)
        return "features done"

    feature_engineering_step()

    print("\nDecorators OK!")
