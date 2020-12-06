//
// Created by Sveta Morkva on 11/28/20.
//

#ifndef KPI_LAB2_FILEUTILS_H
#define KPI_LAB2_FILEUTILS_H

#include <vector>

namespace fileutils {
    std::string getFilenameFromPath(const std::string& path);
    std::vector<std::string> get_file_list(const std::string& path);
    void preprocessData();
};


#endif //KPI_LAB2_FILEUTILS_H
