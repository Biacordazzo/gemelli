import numpy as np
import pandas as pd
from qiime2 import Metadata
from .plugin_setup import plugin
from ._format import TrajectoryFormat, CVFormat, CorrelationFormat


@plugin.register_transformer
def _1(ff: CorrelationFormat) -> pd.DataFrame:
    df = pd.read_csv(str(ff), sep='\t',
                     header=0, index_col=0)
    return df


@plugin.register_transformer
def _2(df: pd.DataFrame) -> CorrelationFormat:
    ff = CorrelationFormat()
    df.to_csv(str(ff), sep='\t', header=True, index=True)
    return


@plugin.register_transformer
def _1(data: pd.DataFrame) -> (TrajectoryFormat):
    ff = TrajectoryFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True, na_rep=np.nan)
    return ff


@plugin.register_transformer
def _2(ff: TrajectoryFormat) -> (pd.DataFrame):
    # with ff.open() as fh:
    return Metadata.load(str(ff)).to_dataframe()


@plugin.register_transformer
def _3(ff: TrajectoryFormat) -> (Metadata):
    # with ff.open() as fh:
    return Metadata.load(str(ff))


@plugin.register_transformer
def _1(data: pd.DataFrame) -> (CVFormat):
    ff = CVFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True, na_rep=np.nan)
    return ff


@plugin.register_transformer
def _2(ff: CVFormat) -> (pd.DataFrame):
    # with ff.open() as fh:
    return Metadata.load(str(ff)).to_dataframe()


@plugin.register_transformer
def _3(ff: CVFormat) -> (Metadata):
    # with ff.open() as fh:
    return Metadata.load(str(ff))
