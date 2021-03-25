import os
import pandas as pd
from subprocess import call
from multiprocessing.dummy import Pool as ThreadPool


WORKPATH = "../"
DATAPATH = WORKPATH + "data/"


def get_exam_data(DATAPATH):
    '''从文献的表格中获取sgRNA序列和切割效应值，即Zscore'''
    allsgRNA = pd.read_excel(DATAPATH+"AP1tables.xlsx", sheet_name="TableS2")
    validsgRNA = pd.read_excel(DATAPATH+"AP1tables.xlsx", sheet_name="TableS3_3")
    validsgRNA["ID"] = validsgRNA["ID"].map(lambda x: ">"+x)
    validSeq = pd.merge(allsgRNA, validsgRNA, on="ID")
    Zscore = validSeq[["ID", "Z"]]
    sgRNAseqs = validSeq[['ID', "Oligo1"]]
    return sgRNAseqs, Zscore


def to_fasta(sgRNAseqs):
    '''把序列转换为标准的fasta格式文件'''
    columns = sgRNAseqs.columns
    seqId = sgRNAseqs[columns[0]].values
    seq = sgRNAseqs[columns[1]].values
    with open(DATAPATH + "sgRNAseqs.fa", "w") as f:
        for n, s in map(list, zip(seqId, seq)):
            f.write(n + "\n")
            f.write(s + "\n")


def align_to_genome(fastafile):
    '''获取sgRNA序列在基因组的对应位置，生成每一条序列所对应的txt文件'''
    file_save_path = DATAPATH+"genome_location\\"
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)

    def submitjob(seq, label):
        '''调用shell命令从网站获取序列对应的基因组位置信息'''
        cmd = "wget https://crispr.dbcls.jp/detail/en/hg19/%s.txt.download\
        -O %s.txt" % (seq.strip()[5:], file_save_path+label)
        call(cmd, shell=True)
    with open(DATAPATH + "sgRNAseqs.fa") as f:
        lines = f.readlines()
    seqs = [line for line in lines if not line.startswith(">")]
    seqIds = [line for line in lines if line.startswith(">")]
    inputs = list(zip(seqs, seqIds))
    pool = ThreadPool()
    pool.starmap(submitjob, inputs)
    pool.close()
    pool.join()


def merge_location(file_save_path):
    '''从文件夹中读取所有文件，并生成所有的序列的位置文件信息'''
    with open(DATAPATH+"genome_loc.txt", "wa") as f:
        for file in os.listdir(file_save_path):
            domain = os.path.abspath(file_save_path)
            filename = os.path.join(domain, file)
            with open(filename) as g:
                lines = g.readlines()
                validline = lines[-2]
                vd = validline.split("\t")
            f.write("%s\t%s\t%s\t%s\n" % (file.split(".")[0], vd[0], vd[2], vd[3]))


def expandseq(flank):
    '''
    flank为需要扩增的侧翼的长度，序列的最终长度为2*flank
    '''
    OriginalFile = pd.read_csv(DATAPATH+"genome_loc.txt", sep="\t", names=["ID", "chromosome", "start", "end"])
    OriginalFile["middle"] = (OriginalFile["start"] + OriginalFile["end"]) // 2
    OriginalFile["new_start"] = OriginalFile["middle"] - flank
    OriginalFile["new_end"] = OriginalFile["middle"] + flank
    finalSeq = OriginalFile[["ID", "chromosome", "new_start", "new_end"]]
    return finalSeq


def main():
    sgRNAseqs, Zscore = get_exam_data(DATAPATH)
    to_fasta(sgRNAseqs)
    align_to_genome(DATAPATH + "sgRNAseqs.fa")
    merge_location(DATAPATH+"genome_location\\")
    flank = 10
    final_seq = expandseq(flank)
    validSeq = pd.merge(final_seq, Zscore, on="ID")
    validSeq = validSeq.sort_values(by="Z", ascending=False)
    validSeq.to_csv(DATAPATH+"AP1sgRNA%d_780.csv"% 2*flank, index=False, header=False)
    # 下边输出文件的后缀命中的序列长度应该为tobed函数中长度的一倍
    validSeq[["chromosome", "new_start", "new_end"]].to_csv(DATAPATH+"AP1sgRNA%dbp.bed" % 2*flank, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
