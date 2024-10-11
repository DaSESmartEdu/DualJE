import argparse

def parse_args():
    parser = argparse.ArgumentParser('Process the data and parameters')

    #data setting
    parser.add_argument(
           '--data', type=str, default='../data', help='data path'
        )

    parser.add_argument(
           '--API_repeated', type=bool, default=False, 
           help='Is the APIset repeated?'
        )
    parser.add_argument(
           '--API_emb_dict', type=str, default='./vocab/API_emb_dict.pickle', 
           help='API embedding dict'
        )
    
    parser.add_argument(
           '--title_dict', type=str, default='./vocab/title_vocab.json', 
           help='title dict'
        )
    parser.add_argument(
           '--log_path', type=str, default='./log/model_log.log', 
           help='log path'
        )
    
    parser.add_argument(
            '--n_APIs', type=int, default=5,
            help='number of API sets'
        )

    parser.add_argument(
            '--shuffle', type=bool, default=False,
            help='shuffle'
        )
    
    parser.add_argument(
            '--num_workers', type=int, default=4,
            help='num_workers'
        )

    #model files path
    parser.add_argument(
           '--primal_model', type=str, default='./model_files/primal/primal_task.pt', 
           help='primal task'
        )
    
    parser.add_argument(
           '--dual_model', type=str, default='./model_files/dual/dual_task_dualmodel.pt', 
           help='dual task'
        )
    
    parser.add_argument(
            '--JM', type=str, default='./model_files/JM/JM.pt',
            help='JM')
    
    parser.add_argument(
            '--EM', type=str, default='./model_files/EM/EM.pth',
            help='EM path')


    #model setting   
    parser.add_argument(
            '--alpha', type=float, default=0.01,
            help='alpha'
        ) 
    parser.add_argument(
            '--beta', type=float, default=0.1,
            help='beta'
        ) 
    parser.add_argument(
            '--epochs', type=int, default=200,
            help='upper epoch limit'
        )

    parser.add_argument(
            '--model-save-path', type=str,  default='./model_files/DualJE_model.pt',
            help='the saving path of model')

    parser.add_argument(
            '--dropout', type=float, default=0.2,
            help='dropout applied to layers (0 = no dropout)'
        )

    parser.add_argument(
            '--n_layers', type=int, default=2,
            help='number of layers'
        )
    
    parser.add_argument(
           '--APIsets_emsize', type=int, default=200,
           help='size of API embeddings'
        )

    parser.add_argument(
           '--emb_size', type=int, default=32,
           help='size of title embeddings( title and APIs type)'
        )

    parser.add_argument(
            '--hidden_size', type=int, default=256,
            help='number of hidden units per layer'
        )
    
    parser.add_argument(
            '--lr', type=float, default=1e-4,
            help='initial learning rate'
        )

    parser.add_argument(
            '--weight_decay', type=float, default=1e-5,
            help='L2'
        )
    
    parser.add_argument(
            '--batch_size', type=int, default=32, metavar='N',
            help='batch size'
        )
    
    parser.add_argument(
            '--seq_len', type=int, default=30,
            help='sequence length'
        )
    
    parser.add_argument(
            '--bidirectional', type=bool, default=True,
            help='wheather bidirectional'
        )
    
    parser.add_argument(
            '--clip', type=float, default=0.01,
            help='gradient clipping'
        )
   
    parser.add_argument(
            '--latent_size', type=int, default=512,
            help='APIs latent size'
        )  
    
    parser.add_argument(
            '--seed', type=int, default=2024,
            help='random seed')

    
    args = parser.parse_args()
    return args