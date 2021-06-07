import subprocess
from subprocess import Popen
import os
import sys
import re
import argparse
import numpy as np
import six, sys, subprocess
import random
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import tensorflow as tf
import time
start = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--inputs', default='sample_inputs/2zzm-B.fasta', type=str, help='Path to input file in fasta format, accept multiple sequences as well in fasta format; default = ''sample_inputs/single_seq.fasta''\n', metavar='')
parser.add_argument('--outputs',default='outputs/', type=str, help='Path to output files; SPOT-RNA outputs at least three files .ct, .bpseq, and .prob files; default = ''outputs/\n', metavar='')
parser.add_argument('--window_size',default=400, type=int, help='Specify the sliding window size; default = 400\n', metavar='')
parser.add_argument('--window_overlap',default=100, type=int, help='Specify the sliding window overlap; default = 100\n', metavar='')
args = parser.parse_args()

base_path = os.path.dirname(os.path.realpath(__file__))
venvPath = os.path.join('.venv/bin/python')
spotrnaPath = os.path.join('SPOT-RNA.py')

def FastaMLtoSL(inputfile):
	inFile  = inputfile
	#outFile = inFile + ".out" 
	outFile = inFile

	print(">> Opening FASTA file...")
	# Reads sequence file list and stores it as a string object. Safely closes file:
	try:	
		with open(inFile,"r") as newFile:
			sequences = newFile.read()
			sequences = re.split("^>", sequences, flags=re.MULTILINE) # Only splits string at the start of a line.
			del sequences[0] # The first fasta in the file is split into an empty empty element and and the first fasta
							 # Del removes this empty element.
			newFile.close()
	except IOError:
		print("Failed to open " + inFile)
		exit(1)

	print(">> Converting FASTA file from multiline to single line and writing to file.")
	# Conversts multiline fasta to single line. Writes new fasta to file.
	try:	
		with open(outFile,"w") as newFasta:
			for fasta in sequences:
				try:
					header, sequence = fasta.split("\n", 1) # Split each fasta into header and sequence.
				except ValueError:
					print(fasta)
				header = ">" + header + "\n" # Replace ">" lost in ">" split, Replace "\n" lost in split directly above.
				sequence = sequence.replace("\n","") + "\n" # Replace newlines in sequence, remember to add one to the end.
				newFasta.write(header + sequence)
			newFasta.close()
	except IOError:
		print("Failed to open " + inFile)
		exit(1)

	print(">> Done!")

	return

# ------------- one hot encoding of RNA sequences -----------------#
def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])

    return feat


def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)


def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)


def get_data(seq):
    seq_len = len(seq)
    one_hot_feat = one_hot(seq)
    #print(one_hot_feat[-1])
    zero_mask = z_mask(seq_len)[None, :, :, None]
    label_mask = l_mask(one_hot_feat, seq_len)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    #out = true_output

    return seq_len, [i for i in (feature.astype(float)).flatten()], [i for i in zero_mask.flatten()], [i for i in label_mask.flatten()], [i for i in label_mask.flatten()]

def _int64_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfr_files(all_seq, base_path, input_file):

    print('\nPreparing tfr records file for SPOT-RNA:')
    path_tfrecords = os.path.join(base_path, 'input_tfr_files', input_file+'.tfrecords')
    with open(all_seq) as file:
        input_data = [line.strip() for line in file.read().splitlines() if line.strip()]

    count = int(len(input_data)/2)

    ids = [input_data[2*i][1:].strip() for i in range(count)]
    
    with tf.io.TFRecordWriter(path_tfrecords) as writer:
        for i in tqdm(range(len(ids))):
            name     = input_data[2*i].replace(">", "") 
            sequence = input_data[2*i+1].replace(" ", "").upper().replace("T", "U")
            #print(sequence[-1])
            
            #print(len(sequence), name)                
            seq_len, feature, zero_mask, label_mask, true_label = get_data(sequence)

            example = tf.train.Example(features=tf.train.Features(feature={'rna_name': _bytes_feature(name),
                                                                           'seq_len': _int64_feature(seq_len),
                                                                           'feature': _float_feature(feature),
                                                                           'zero_mask': _float_feature(zero_mask),
                                                                           'label_mask': _float_feature(label_mask),
                                                                           'true_label': _float_feature(true_label)}))

            writer.write(example.SerializeToString())

    writer.close()

# ----------------------- hair pin loop assumption i - j < 2 --------------------------------#
def hair_pin_assumption(pred_pairs):
    pred_pairs_all = [i[:2] for i in pred_pairs]
    bad_pairs = []
    for i in pred_pairs_all:
        if abs(i[0] - i[1]) < 3:
            bad_pairs.append(i)
    return bad_pairs

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]],sequence[i[1]]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]],sequence[i[1]]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t

# ----------------------- find multiplets pairs--------------------------------#
def multiplets_pairs(pred_pairs):

    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)
        
    return temp3

def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs)+len(save_multiplets)
    #print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets
        
def output_mask(seq, NC=True):
    if NC:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
    else:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
    mask = np.zeros((len(seq), len(seq)))
    for i, I in enumerate(seq):
        for j, J in enumerate(seq):
            if str(I) + str(J) in include_pairs:
                mask[i, j] = 1
    return mask

def ct_file_output(pairs, seq, id, save_result_path):

    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]] = int(I[1]) + 1
        col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, str(id))+'.ct', (temp), delimiter='\t\t', fmt="%s", header=str(len(seq)) + '\t\t' + str(id) + '\t\t' + 'SPOT-RNA output\n' , comments='')

    return

def bpseq_file_output(pairs, seq, id, save_result_path):

    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    #col3 = np.arange(0, len(seq), 1)
    #col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]] = int(I[1]) + 1
        col5[I[1]] = int(I[0]) + 1
    #col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col5))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, str(id))+'.bpseq', (temp), delimiter=' ', fmt="%s", header='#' + str(id) , comments='')

    return

def lone_pair(pairs):
    lone_pairs = []
    pairs.sort()
    for i, I in enumerate(pairs):
        if ([I[0] - 1, I[1] + 1] not in pairs) and ([I[0] + 1, I[1] - 1] not in pairs):
            lone_pairs.append(I)

    return lone_pairs

def prob_to_secondary_structure(ensemble_outputs, label_mask, seq, name, args, base_path):
    #save_result_path = 'outputs'
    Threshold = 0.335
    test_output = ensemble_outputs
    mask = output_mask(seq)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]
    y_pred = np.multiply(y_pred, mask)

    tri_inds = np.triu_indices(y_pred.shape[0], k=1)

    out_pred = y_pred[tri_inds]
    outputs = out_pred[:, None]
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([seq[tri_inds[0][j]], seq[tri_inds[1][j]]])] for j in
                 range(tri_inds[0].shape[0])]

    outputs_T = np.greater_equal(outputs, Threshold)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    pred_pairs = [i[:2] for i in pred_pairs]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred)
    
    watson_pairs, wobble_pairs, noncanonical_pairs = type_pairs(pred_pairs, seq)
    lone_bp = lone_pair(pred_pairs)

    tertiary_bp = save_multiplets + noncanonical_pairs + lone_bp
    tertiary_bp = [list(x) for x in set(tuple(x) for x in tertiary_bp)]

    str_tertiary = []
    for i,I in enumerate(tertiary_bp):
        if i==0: 
            str_tertiary += ('(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')   
        
    tertiary_bp = ''.join(str_tertiary) 

    if args.outputs=='outputs/':
        output_path = os.path.join(base_path, args.outputs)
    else:
        output_path = args.outputs

    ct_file_output(pred_pairs, seq, name, output_path)
    bpseq_file_output(pred_pairs, seq, name, output_path)
    np.savetxt(output_path + '/'+ name +'.prob', y_pred, delimiter='\t')
    
    # if args.plots:
    #     try:
    #         subprocess.Popen(["java", "-cp", base_path + "/utils/VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', output_path + name + '.ct', '-o', output_path + name + '_radiate.png', '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    #         subprocess.Popen(["java", "-cp", base_path + "/utils/VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', output_path + name + '.ct', '-o', output_path + name + '_line.png', '-algorithm', 'line', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    #     except:
    #         print('\nUnable to generate 2D plots;\nplease refer to "http://varna.lri.fr/" for system requirments to use VARNA')	

    # if args.motifs:
    #     try:
    #         os.chdir(output_path)
    #         p = subprocess.Popen(['perl', base_path + '/utils/bpRNA-master/bpRNA.pl', name + '.bpseq'])
    #     except:
    #         print('\nUnable to run bpRNA script;\nplease refer to "https://github.com/hendrixlab/bpRNA/" for system requirments to use bpRNA')
    #     os.chdir('../')
    return

def merged_prob_to_secondary_structure(fragments_probs, seq, name, args, base_path):
    Threshold = 0.335
    tri_inds = np.triu_indices(fragments_probs.shape[0], k=1)
    out_pred = fragments_probs[tri_inds]
    outputs = out_pred[:, None]
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([seq[tri_inds[0][j]], seq[tri_inds[1][j]]])] for j in range(tri_inds[0].shape[0])]
    outputs_T = np.greater_equal(outputs, Threshold)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    pred_pairs = [i[:2] for i in pred_pairs]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, fragments_probs)
    watson_pairs, wobble_pairs, noncanonical_pairs = type_pairs(pred_pairs, seq)
    lone_bp = lone_pair(pred_pairs)
    tertiary_bp = save_multiplets + noncanonical_pairs + lone_bp
    tertiary_bp = [list(x) for x in set(tuple(x) for x in tertiary_bp)]
    str_tertiary = []
    for i,I in enumerate(tertiary_bp):
        if i==0: 
            str_tertiary += ('(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]+1) + ',' + str(I[1]+1) + '):color=""#FFFF00""')   
    tertiary_bp = ''.join(str_tertiary)
    output_path = args.outputs
    prob_file_name = name + '_merged' + '.prob'
    prob_file_path = os.path.join(output_path, prob_file_name)
    ct_file_output(pred_pairs, seq, name, output_path)
    bpseq_file_output(pred_pairs, seq, name, output_path)
    np.savetxt(prob_file_path, fragments_probs, delimiter='\t')
    ct_file_path = os.path.join(output_path, str(name) + '.ct')
    radiate_file_path = os.path.join(output_path, str(name) + '_radiate.png')
    line_file_path = os.path.join(output_path, str(name) + '_line.png')
    try:
        subprocess.Popen(["java", "-cp", base_path + "/utils/VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', ct_file_path, '-o', radiate_file_path, '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        subprocess.Popen(["java", "-cp", base_path + "/utils/VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', ct_file_path, '-o', line_file_path, '-algorithm', 'line', '-resolution', '8.0', '-bpStyle', 'lw', '-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    except:
        print('\nUnable to generate 2D plots;\nplease refer to "http://varna.lri.fr/" for system requirments to use VARNA')	
    return

def sigmoid(x):
    return 1/(1+np.exp(-np.array(x, dtype=np.float128)))

def SPOT_RNA(inputs, outputs):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    FastaMLtoSL(inputs)

    base_path = os.path.dirname(os.path.realpath(__file__))
    input_file = os.path.basename(inputs)

    create_tfr_files(inputs, base_path, input_file)

    with open(inputs) as file:
        input_data = [line.strip() for line in file.read().splitlines() if line.strip()]

    count = int(len(input_data)/2)

    ids = [input_data[2*i].replace(">", "") for i in range(count)]
    sequences = {}
    for i,I in enumerate(ids):
        sequences[I] = input_data[2*i+1].replace(" ", "").upper().replace("T", "U")

    # os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    NUM_MODELS = 1

    test_loc = [os.path.join(base_path, 'input_tfr_files', input_file+'.tfrecords')]

    outputs = {}
    mask = {}

    for MODEL in range(NUM_MODELS):

        # if args.gpu==-1:
        config = tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=24)
        # else:
        #     config = tf.compat.v1.ConfigProto()
        #     config.allow_soft_placement=True
        #     config.log_device_placement=False
            
        print('\nPredicting for SPOT-RNA model '+str(MODEL))
        with tf.compat.v1.Session(config=config) as sess:
            saver = tf.compat.v1.train.import_meta_graph(os.path.join(base_path, 'SPOT-RNA-models', 'model' + str(MODEL) + '.meta'))
            saver.restore(sess,os.path.join(base_path, 'SPOT-RNA-models', 'model' + str(MODEL)))
            graph = tf.compat.v1.get_default_graph()
            init_test =  graph.get_operation_by_name('make_initializer_2')
            tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')
            name_tensor = graph.get_tensor_by_name('tensors_2/component_0:0')
            RNA_name = graph.get_tensor_by_name('IteratorGetNext:0')
            label_mask = graph.get_tensor_by_name('IteratorGetNext:4')
            sess.run(init_test,feed_dict={name_tensor:test_loc})
            
            pbar = tqdm(total = count)
            while True:
                try:        
                    out = sess.run([tmp_out,RNA_name,label_mask],feed_dict={'dropout:0':1})
                    out[1] = out[1].decode()
                    mask[out[1]] = out[2]
                    
                    if MODEL == 0:
                        outputs[out[1]] = [sigmoid(out[0])]
                    else:
                        outputs[out[1]].append(sigmoid(out[0]))
                    #print('RNA name: %s'%(out[1]))
                    pbar.update(1)
                except:
                    break
            pbar.close()
        tf.compat.v1.reset_default_graph()


    RNA_ids = [i for i in list(outputs.keys())]
    ensemble_outputs = {}

    print('\nPost Processing and Saving Output')
    for i in RNA_ids:
        ensemble_outputs[i] = np.mean(outputs[i],0)
        prob_to_secondary_structure(ensemble_outputs[i], mask[i], sequences[i], i, args, base_path)

    print('\nFinished!')
    end = time.time()
    print('\nProcesssing Time {} seconds'.format(end - start))

def run():

    FastaMLtoSL(args.inputs)
    for record in SeqIO.parse(args.inputs, "fasta"):
        sequence_id = record.id
        sequence = str(record.seq)
        length = len(sequence)
        window_size = args.window_size
        window_overlap = args.window_overlap
        inputs = args.inputs
        outputs = args.outputs

        current = 0
        ranges = []
        while True:
            ranges.append((current, current + window_size))
            if current + window_size > length:
                break
            else:
                current = current + window_size - window_overlap

        processes = []
        for ix, r in enumerate(ranges):
            sub_sequence_id = sequence_id + '_part_' + str(ix)
            sub_sequence = sequence[r[0]:r[1]]
            sub_record = SeqRecord(Seq(sub_sequence), sub_sequence_id, '', '')
            base_name = os.path.splitext(os.path.basename(inputs))[0]
            extension = os.path.splitext(os.path.basename(inputs))[1]
            sub_file_name = sub_sequence_id + extension
            sub_file_path = os.path.join(outputs, sub_file_name)
            SeqIO.write(
                sub_record, 
                sub_file_path, 
                "fasta"
                )
            try:
                # p = subprocess.Popen([venvPath, spotrnaPath, '--inputs', sub_file_path, '--outputs', outputs], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
                SPOT_RNA(sub_file_path, outputs)
            except Exception as e:
                print('exception', e)
        for p in processes:
            p.join()

        partial_probs_list = []
        for ix, r in enumerate(ranges):
            prob_file_name = sequence_id + '_part_' + str(ix) + '.prob'
            prob_file_path = os.path.join(outputs, prob_file_name)
            probs = np.loadtxt(prob_file_path, delimiter="\t")
            partial_probs_list.append(probs)

        full_probs = np.zeros((length, length))
        probs_length = len(partial_probs_list)
        for ix in range(probs_length):
            current_probs = partial_probs_list[ix]
            next_probs = None if ix == (probs_length - 1) else partial_probs_list[ix + 1]
            start_pos_1 = ix * (window_size - window_overlap)
            end_pos = min((ix + 1) * (window_size) - (ix * window_overlap), length)
            for i in range(start_pos_1, end_pos):
                for j in range(start_pos_1, end_pos):
                    if (full_probs[i, j] == 0):
                        full_probs[i, j] = current_probs[i - start_pos_1][j - start_pos_1]
                    else:
                        full_probs[i, j] = ((current_probs[i - start_pos_1][j - start_pos_1] + full_probs[i, j])/2)
            start_pos_2 = min((ix + 1) * (window_size - window_overlap), length)
            end_pos = min((ix + 1) * (window_size) - (ix * window_overlap), length)
            for i in range(start_pos_2, end_pos):
                for j in range(start_pos_2, end_pos):
                    if (next_probs is not None):
                        full_probs[i, j] = current_probs[i - start_pos_1][j - start_pos_1] + next_probs[i -  start_pos_1 - (window_size - window_overlap)][j - start_pos_1 - (window_size - window_overlap)]
                    else:
                        full_probs[i, j] = current_probs[i - start_pos_1][j - start_pos_1]
        merged_prob_to_secondary_structure(full_probs, sequence, sequence_id, args, base_path)


if __name__ == "__main__":
    run()

