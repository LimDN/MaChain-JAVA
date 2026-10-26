package machain.output;

import machain.network.Network;
import machain.data.DataSet;
import machain.core.TrainingResult;

public interface OutputModule {
    String getName();
    void exportTrainingResult(TrainingResult result, String outputPath) throws Exception;
    void exportPredictions(Network network, DataSet dataSet, String outputPath) throws Exception;
    void exportModelInfo(Network network, String outputPath) throws Exception;
}
