package com.kse.bigdata.main;

import com.kse.bigdata.file.SourceFileMerger;

/**
 * Created by user on 2014-12-02.
 */
public class Driver {

    public static void main(String[] args){
        SourceFileMerger sourceFileMerger = new SourceFileMerger("C:\\Users\\user\\Downloads", "result.csv");
        sourceFileMerger.mergeFiles();
    }
}
