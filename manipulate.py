import array
import random
import functools
import signal
import multiprocessing


class SampleManipulator(object):
    def __init__(self, bytez):
        self.bytez = bytez
        self.min_append_log2 = 5
        self.max_append_log2 = 8

    def __random_length(self):
        return 2 ** random.randint(self.min_append_log2, self.max_append_log2)

    def change_duration(self, seed=None):
        # random.seed(seed)
        # L = self.__random_length()
        # # choose the upper bound for a uniform distribution in [0,upper]
        # upper = random.randrange(256)
        # # upper chooses the upper bound on uniform distribution:
        # # upper=0 would append with all 0s
        # # upper=126 would append with "printable ascii"
        # # upper=255 would append with any character
        # return self.bytez + bytes([random.randint(0, upper) for _ in range(L)])

        pass

    def change_InBytes(self, seed=None):
        # # add (unused) imports
        # random.seed(seed)
        # binary = lief.PE.parse(self.bytez)
        # # draw a library at random
        # libname = random.choice(list(COMMON_IMPORTS.keys()))
        # funcname = random.choice(list(COMMON_IMPORTS[libname]))
        # lowerlibname = libname.lower()
        # # find this lib in the imports, if it exists
        # lib = None
        # for im in binary.imports:
        #     if im.name.lower() == lowerlibname:
        #         lib = im
        #         break
        # if lib is None:
        #     # add a new library
        #     lib = binary.add_library(libname)
        # # get current names
        # names = set([e.name for e in lib.entries])
        # if not funcname in names:
        #     lib.add_entry(funcname)

        # self.bytez = self.__binary_to_bytez(binary, imports=True)

        # return self.bytez

        pass

    def change_OutBytes(self, seed=None):
        # # rename a random section
        # random.seed(seed)
        # binary = lief.PE.parse(self.bytez)
        # targeted_section = random.choice(binary.sections)
        # targeted_section.name = random.choice(COMMON_SECTION_NAMES)[
        #     :7
        # ]  # current version of lief not allowing 8 chars?

        # self.bytez = self.__binary_to_bytez(binary)

        # return self.bytez
        pass

    def change_TotPkts(self, seed=None):
        # random.seed(seed)
        # binary = lief.PE.parse(self.bytez)
        # new_section = lief.PE.Section(
        #     "".join(chr(random.randrange(ord("."), ord("z"))) for _ in range(6))
        # )

        # # fill with random content
        # upper = random.randrange(256)
        # L = self.__random_length()
        # new_section.content = [random.randint(0, upper) for _ in range(L)]

        # new_section.virtual_address = max(
        #     [s.virtual_address + s.size for s in binary.sections]
        # )
        # # add a new empty section

        # binary.add_section(
        #     new_section,
        #     random.choice(
        #         [
        #             lief.PE.SECTION_TYPES.BSS,
        #             lief.PE.SECTION_TYPES.DATA,
        #             lief.PE.SECTION_TYPES.EXPORT,
        #             lief.PE.SECTION_TYPES.IDATA,
        #             lief.PE.SECTION_TYPES.RELOCATION,
        #             lief.PE.SECTION_TYPES.RESOURCE,
        #             lief.PE.SECTION_TYPES.TEXT,
        #             lief.PE.SECTION_TYPES.TLS_,
        #             lief.PE.SECTION_TYPES.UNKNOWN,
        #         ]
        #     ),
        # )

        # self.bytez = self.__binary_to_bytez(binary)
        # return self.bytez
        pass


ACTION_TABLE = {
    "TotalLengthofFwdPacket": "TotalLengthofFwdPacket",
    "FwdPacketLengthMean": "FwdPacketLengthMean",
    "BwdPackets/s": "BwdPackets/s",
    "SubflowFwdBytes": "SubflowFwdBytes",
}

# ACTION_TABLE = {
#     "FlowDuration": "FlowDuration",
#     "FlowBytes/s": "FlowBytes/s",
#     "FlowPackets/s": "FlowPackets/s",
#     "FwdPackets/s": "FwdPackets/s",
#     "BwdPackets/s": "BwdPackets/s",
# }


def modify(bytez, actions=[], seed=None):
    for action in actions:

        _action = ACTION_TABLE[action]

        def helper(_action, shared_list):
            # TODO: LIEF is chatty. redirect stdout and stderr to /dev/null

            # for this process, change segfault of the child process
            # to a RuntimeEror
            def sig_handler(signum, frame):
                raise RuntimeError

            signal.signal(signal.SIGSEGV, sig_handler)

            bytez = array.array("B", shared_list[:]).tobytes()
            # TODO: LIEF is chatty. redirect output to /dev/null
            if type(_action) is str:
                _action = SampleManipulator(bytez).__getattribute__(_action)
            else:
                _action = functools.partial(_action, bytez)

            # redirect standard out only in this queue
            try:
                shared_list[:] = _action(seed)
            except (RuntimeError, UnicodeDecodeError, TypeError, lief.not_found) as e:
                # some exceptions that have yet to be handled by public release of LIEF
                print("==== exception in child process ===")
                print(e)
                # shared_bytez remains unchanged

        # communicate with the subprocess through a shared list
        # can't use multiprocessing.Array since the subprocess may need to
        # change the size
        manager = multiprocessing.Manager()
        shared_list = manager.list()
        shared_list[:] = bytez  # copy bytez to shared array
        # define process
        p = multiprocessing.Process(target=helper, args=(_action, shared_list))
        p.start()  # start the process
        try:
            p.join(5)  # allow this to take up to 5 seconds...
        except multiprocessing.TimeoutError:  # ..then become petulant
            print("==== timeouterror ")
            p.terminate()

        bytez = array.array(
            "B", shared_list[:]
        ).tobytes()  # copy result from child process

    import hashlib

    m = hashlib.sha256()
    m.update(bytez)
    print("new hash: {}".format(m.hexdigest()))
    return bytez


# def test(bytez):
#     binary = lief.PE.parse(bytez)

#     print("change_duration")
#     manip = SampleManipulator(bytez)
#     bytez2 = manip.overlay_append(bytez)
#     binary2 = lief.PE.parse(bytez2)
#     assert len(binary.overlay) != len(binary2.overlay), "modification failed"

#     print("change_InBytes")
#     manip = SampleManipulator(bytez)
#     bytez2 = manip.imports_append(bytez)
#     # binary2 = lief.PE.parse(bytez2)
#     set1 = set(binary.imported_functions)
#     set2 = set(binary2.imported_functions)
#     diff = set2.difference(set1)
#     print(list(diff))
#     assert len(binary.imported_functions) != len(
#         binary2.imported_functions
#     ), "no new imported functions"

#     print("change_OutBytes")
#     manip = SampleManipulator(bytez)
#     bytez2 = manip.section_rename(bytez)
#     binary2 = lief.PE.parse(bytez2)
#     oldsections = [s.name for s in binary.sections]
#     newsections = [s.name for s in binary2.sections]
#     print(oldsections)
#     print(newsections)
#     assert " ".join(newsections) != " ".join(oldsections), "no modified sections"

#     print("change_TotPkts")
#     manip = SampleManipulator(bytez)
#     bytez2 = manip.section_add(bytez)
#     binary2 = lief.PE.parse(bytez2)
#     oldsections = [s.name for s in binary.sections]
#     newsections = [s.name for s in binary2.sections]
#     print(oldsections)
#     print(newsections)
#     assert len(newsections) != len(oldsections), "no new sections"
