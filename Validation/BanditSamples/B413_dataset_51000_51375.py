def generate_keypair(keypair_file):
    '''generate_keypair is used by some of the helpers that need a keypair.
       The function should be used if the client doesn't have the attribute 
       self.key. We generate the key and return it.

       We use pycryptodome (3.7.2)       

       Parameters
       =========
       keypair_file: fullpath to where to save keypair
    '''

    from Crypto.PublicKey import RSA
    key = RSA.generate(2048)

    # Ensure helper directory exists
    keypair_dir = os.path.dirname(keypair_file)
    if not os.path.exists(keypair_dir):
        os.makedirs(keypair_dir)

    # Save key
    with open(keypair_file, 'wb') as filey:
        filey.write(key.exportKey('PEM'))

    return key