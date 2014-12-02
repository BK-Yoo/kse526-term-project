package com.kse.bigdata.main;

import com.kse.bigdata.file.SourceFileMerger;

/**
 * Created by user on 2014-12-02.
 */
public class Driver {

    public static void main(String[] args){
        //!!!!!!Should change the directories of each file before executing the program!!!!!
        String inputFile = "D:\\BigData_Term_Project\\Data";
        String resultFile = "D:\\BigData_Term_Project\\Merge_Result.csv";
        SourceFileMerger sourceFileMerger = new SourceFileMerger(inputFile, resultFile);
        sourceFileMerger.mergeFiles();
    }
}
