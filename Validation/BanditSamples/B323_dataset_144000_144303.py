def main():
   """
   Simple command-line program for dumping the contents of any managed object.
   """

   args = GetArgs()
   if args.password:
      password = args.password
   else:
      password = getpass.getpass(prompt='Enter password for host %s and '
                                        'user %s: ' % (args.host,args.user))

   context = None
   if hasattr(ssl, '_create_unverified_context'):
      context = ssl._create_unverified_context()
   si = SmartConnect(host=args.host,
                     user=args.user,
                     pwd=password,
                     port=int(args.port),
                     sslContext=context)
   if not si:
       print("Could not connect to the specified host using specified "
             "username and password")
       return -1

   atexit.register(Disconnect, si)

   obj = VmomiSupport.templateOf(args.type)(args.id, si._stub)
   print(json.dumps(obj, cls=VmomiSupport.VmomiJSONEncoder,
                    sort_keys=True, indent=4))