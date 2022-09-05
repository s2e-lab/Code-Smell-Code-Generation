def main():
   """
   Simple command-line program for powering on virtual machines on a system.
   """

   args = GetArgs()
   if args.password:
      password = args.password
   else:
      password = getpass.getpass(prompt='Enter password for host %s and user %s: ' % (args.host,args.user))

   try:
      vmnames = args.vmname
      if not len(vmnames):
         print("No virtual machine specified for poweron")
         sys.exit()

      context = None
      if hasattr(ssl, '_create_unverified_context'):
         context = ssl._create_unverified_context()
      si = SmartConnect(host=args.host,
                        user=args.user,
                        pwd=password,
                        port=int(args.port),
                        sslContext=context)
      if not si:
         print("Cannot connect to specified host using specified username and password")
         sys.exit()

      atexit.register(Disconnect, si)

      # Retreive the list of Virtual Machines from the inventory objects
      # under the rootFolder
      content = si.content
      objView = content.viewManager.CreateContainerView(content.rootFolder,
                                                        [vim.VirtualMachine],
                                                        True)
      vmList = objView.view
      objView.Destroy()

      # Find the vm and power it on
      tasks = [vm.PowerOn() for vm in vmList if vm.name in vmnames]

      # Wait for power on to complete
      WaitForTasks(tasks, si)

      print("Virtual Machine(s) have been powered on successfully")
   except vmodl.MethodFault as e:
      print("Caught vmodl fault : " + e.msg)
   except Exception as e:
      print("Caught Exception : " + str(e))