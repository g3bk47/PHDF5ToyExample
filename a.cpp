
#include "hdf5.h"
#include <string>
#include <vector>
#include <iostream>
#include <memory>

class ParallelHDF5File
{
    public:
        ParallelHDF5File() : file(-1), plist_id(-1), mpi_size(-1) {}

        ParallelHDF5File(const ParallelHDF5File&) = delete;

        ParallelHDF5File(ParallelHDF5File&& rhs) :
            file(rhs.file), plist_id(rhs.plist_id), mpi_size(rhs.mpi_size)
        {
            rhs.file = -1;
            rhs.plist_id = -1;
        }

        ~ParallelHDF5File(){close();};

        ParallelHDF5File& operator=(ParallelHDF5File&& rhs)
        {
            if (&rhs == this)
                return *this;

            file = rhs.file;
            rhs.file = -1;
            plist_id = rhs.plist_id;
            rhs.plist_id = -1;
            mpi_size = rhs.mpi_size;
            return *this;
        }

        ParallelHDF5File& operator=(const ParallelHDF5File& rhs) = delete;

        ParallelHDF5File(const std::string& name, MPI_Comm& comm, MPI_Info& info)
            : file(-1), plist_id(-1), mpi_size(-1)
        {
            open(name, comm, info);
        }

        void open(const std::string& name, MPI_Comm& comm, MPI_Info& info)
        {
            close();

            MPI_Comm_size(comm, &mpi_size);

            // create file in parallel
            plist_id = H5Pcreate(H5P_FILE_ACCESS);
            if (plist_id == -1)
            {
                std::cerr<<"ERROR: cannot open file "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            herr_t err = H5Pset_fapl_mpio(plist_id, comm, info);
            if (err == -1)
            {
                std::cerr<<"ERROR: cannot set mpi i/o for "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            file = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
            if (file == -1)
            {
                std::cerr<<"ERROR: cannot open hdf5 file for "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // creates mpi_size groups called basename[mpi_rank]
        auto create_group_per_rank(const std::string& basename)
        {
            std::shared_ptr<std::vector<hid_t>> p(new std::vector<hid_t>(mpi_size, -1),
                    [](std::vector<hid_t>* p)
                    {
                        if (p == nullptr)
                            return;
                        for(auto& e : *p)
                        {
                            if (e != -1)
                            {
                                herr_t err = H5Gclose(e);
                                if (err == -1)
                                {
                                    std::cerr<<"ERROR: cannot close group"<<std::endl;
                                    std::exit(EXIT_FAILURE);
                                }
                                err = -1;
                            }
                        }
                        delete p;
                    });

            auto& groups = *p;

            for(int i=0; i!=mpi_size; ++i)
            {
                std::string groupname = basename + std::to_string(i);

                // all ranks participate in the creation of each group
                groups[i] = H5Gcreate (file, groupname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (groups[i] == -1)
                {
                    std::cerr<<"ERROR: cannot create group"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
            return p;
        }

        void close()
        {
            if (plist_id != -1)
                H5Pclose(plist_id);
            if (file != -1)
                H5Fclose(file);
            plist_id = -1;
            file = -1;
        }

        // write an int attribute to a specific group (all ranks have to call this)
        void write_attribute_to_group(hid_t group, const std::string& name, int value)
        {
            hid_t space = H5Screate(H5S_SCALAR);
            if (space == -1)
            {
                std::cerr<<"ERROR: cannot create space"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hid_t attr = H5Acreate (group, name.c_str(), H5T_STD_I64BE, space, H5P_DEFAULT, H5P_DEFAULT);
            if (attr == -1)
            {
                std::cerr<<"ERROR cannot create attribute"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            herr_t err = H5Awrite (attr, H5T_NATIVE_INT, &value);
            if (err == -1)
            {
                std::cerr<<"ERROR: cannot write attribute"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            err = H5Aclose (attr);
            if (err == -1)
            {
                std::cerr<<"ERROR: cannot close attribute"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            err = H5Sclose (space);
            if (err == -1)
            {
                std::cerr<<"ERROR: cannot close space"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // for each group, create a dataset with the size required by each rank (can be different among ranks), has to be called by all ranks for all datasets
        auto create_datasets_for_groups(const std::shared_ptr<std::vector<hid_t>>& Groups, const std::string& name, const std::vector<hsize_t>& length)
        {
            std::shared_ptr<std::vector<std::pair<hid_t,hid_t>>> datasets
                (new std::vector<std::pair<hid_t,hid_t>>(mpi_size, std::pair<hid_t,hid_t>(-1,-1)),
                 [](std::vector<std::pair<hid_t,hid_t>>* p)
                 {
                    if (p==nullptr)
                        return;
                    for (auto& e : *p)
                    {
                        hid_t& dset = e.first;
                        if (dset != -1)
                        {
                            herr_t err = H5Dclose (dset);
                            if (err == -1)
                            {
                                std::cerr<<"ERROR: cannot close dataspace"<<std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                            dset = -1;
                        }

                        hid_t& space = e.second;
                        if (space != -1)
                        {
                            herr_t err = H5Sclose (space);
                            if (err == -1)
                            {
                                std::cerr<<"ERROR: cannot close space"<<std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                            space = -1;
                        }
                    }
                    delete p;
                 }
                );


            for(int i=0; i!=mpi_size; ++i)
            {
                hsize_t dims[2]={length[i],static_cast<hsize_t>(1)};
                hid_t space = H5Screate_simple (1, dims, NULL);
                if (space == -1)
                {
                    std::cerr<<"ERROR: cannot close space"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                hid_t dset = H5Dcreate ((*Groups)[i], name.c_str(), H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (dset == -1)
                {
                    std::cerr<<"ERROR: cannot close space"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                (*datasets)[i] = std::make_pair(dset, space);
            }
            return datasets;
        }

        // write the actual data to the file. This can be called by individual ranks
        void write_array_to_dataset(const std::shared_ptr<std::vector<std::pair<hid_t,hid_t>>>& datasets, int rank, const std::vector<int>& data)
        {
            herr_t ret = H5Dwrite ((*datasets)[rank].first, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
            if (ret == -1)
            {
                    std::cerr<<"ERROR: cannot write to dataspace"<<std::endl;
                    std::exit(EXIT_FAILURE);
            }
        }

        hid_t file;
        hid_t plist_id;
        int mpi_size;
};


int main (int argc, char **argv)
{
    int mpi_size, mpi_rank;
    MPI_Comm comm  = MPI_COMM_WORLD;
    MPI_Info info  = MPI_INFO_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    {
        ParallelHDF5File File("SDS_row.h5", comm, info);

        // collective call: create one group for each rank
        auto Groups = File.create_group_per_rank("/testgroup");

        // collective call: create an attribute for each group containing the rank
        for (int i=0; i!=mpi_size; ++i)
            File.write_attribute_to_group((*Groups)[i], "attr"+std::to_string(i), i);

        std::vector<hsize_t> individualLengths(mpi_size);
        hsize_t localLength = 10 + mpi_rank; // each rank writes a different amount of ints

        // since all ranks have to call the dataset creation, make the dataset length required by each rank known to each other rank
        MPI_Allgather(&localLength, 1, MPI_UNSIGNED_LONG, individualLengths.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
        // collective call to create the datasets
        auto datasets = File.create_datasets_for_groups(Groups, "mesh", individualLengths);

        std::vector<int> data(10+mpi_rank, mpi_rank);
        //write the actual data in parallel (however, non-parallel in practice)
        for(int i=0; i!=mpi_size; ++i)
            if (mpi_rank == i)
                File.write_array_to_dataset(datasets, i, data);

    }
    MPI_Finalize();
}
