"""
Metadata processor for combining and transforming GEO and SRA metadata using pysradb
"""

import pandas as pd
import time
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from pysradb.sraweb import SRAweb
import requests
from bs4 import BeautifulSoup


class MetadataProcessor:
    """Process and combine metadata from GEO and SRA sources using pysradb"""
    
    def __init__(self):
        self.sraweb = SRAweb()
    
    def process_geo_series(self, geo_series: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process a complete GEO series, retrieving both GEO and SRA metadata
        
        Args:
            geo_series: GEO series identifier (e.g., 'GSE212963')
            output_file: Optional path to save CSV output
            
        Returns:
            DataFrame with combined metadata, one row per SRA file
        """
        print(f"Processing GEO series: {geo_series}")
        
        # Step 1: Get series-level metadata from GEO webpage
        print("Retrieving series metadata...")
        series_metadata = self._get_geo_series_metadata(geo_series)
        
        # Step 2: Map GEO series to SRA project
        print("Mapping to SRA project...")
        try:
            srp_df = self.sraweb.gse_to_srp(geo_series)
            if srp_df.empty:
                raise ValueError(f"No SRA project found for {geo_series}")
            
            srp_id = srp_df['study_accession'].iloc[0]
            print(f"Found SRA project: {srp_id}")
            
        except Exception as e:
            print(f"Error mapping to SRA: {e}")
            print("This GEO series may not have associated SRA data.")
            print("Processing as GEO-only series...")
            
            # For GEO-only series, extract sample information from GEO
            return self._process_geo_only_series(geo_series, series_metadata, output_file)
        
        # Step 3: Get comprehensive SRA metadata
        print("Retrieving SRA metadata...")
        try:
            sra_metadata = self.sraweb.sra_metadata(srp_id)
            print(f"Found {len(sra_metadata)} SRA records")
            
        except Exception as e:
            print(f"Error getting SRA metadata: {e}")
            # Fallback to srp_to_srr
            try:
                sra_metadata = self.sraweb.srp_to_srr(srp_id)
                print(f"Found {len(sra_metadata)} SRA records (fallback method)")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return pd.DataFrame()
        
        if sra_metadata.empty:
            print("No SRA metadata found")
            return pd.DataFrame()
        
        # Step 4: Combine metadata
        print("Combining metadata...")
        result_df = self._combine_metadata(series_metadata, sra_metadata)
        
        # Step 5: Add file URLs and additional info
        result_df = self._add_file_information(result_df)
        
        # Step 6: Save to file if requested
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        print(f"Processing complete. Generated {len(result_df)} rows of file metadata.")
        return result_df
    
    def _get_geo_series_metadata(self, geo_series: str) -> Dict:
        """
        Get series metadata from GEO webpage
        
        Args:
            geo_series: GEO series identifier
            
        Returns:
            Dictionary of series metadata
        """
        metadata = {
            'series_id': geo_series,
            'title': '',
            'summary': '',
            'organism': '',
            'platform': '',
            'submission_date': '',
            'last_update_date': ''
        }
        
        try:
            series_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_series}"
            response = requests.get(series_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('td', string='Title')
            if title_elem and title_elem.find_next_sibling('td'):
                metadata['title'] = title_elem.find_next_sibling('td').get_text(strip=True)
            
            # Extract summary
            summary_elem = soup.find('td', string='Summary')
            if summary_elem and summary_elem.find_next_sibling('td'):
                metadata['summary'] = summary_elem.find_next_sibling('td').get_text(strip=True)
            
            # Extract organism
            organism_elem = soup.find('td', string='Organism')
            if organism_elem and organism_elem.find_next_sibling('td'):
                metadata['organism'] = organism_elem.find_next_sibling('td').get_text(strip=True)
            
            # Extract submission date
            date_elem = soup.find('td', string='Submission date')
            if date_elem and date_elem.find_next_sibling('td'):
                metadata['submission_date'] = date_elem.find_next_sibling('td').get_text(strip=True)
            
            # Extract last update
            update_elem = soup.find('td', string='Last update date')
            if update_elem and update_elem.find_next_sibling('td'):
                metadata['last_update_date'] = update_elem.find_next_sibling('td').get_text(strip=True)
            
            # Extract platform info
            platform_links = soup.find_all('a', href=lambda x: x and 'acc.cgi?acc=GPL' in x)
            if platform_links:
                platforms = [link.get_text(strip=True) for link in platform_links]
                metadata['platform'] = ', '.join(platforms)
                
        except Exception as e:
            print(f"Warning: Could not retrieve series metadata from GEO webpage: {e}")
            
        return metadata
    
    def _process_geo_only_series(self, geo_series: str, series_metadata: dict, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process a GEO series that doesn't have SRA data by extracting sample information from GEO
        
        Args:
            geo_series: GEO series identifier
            series_metadata: Series-level metadata already extracted
            output_file: Optional path to save CSV output
            
        Returns:
            DataFrame with GEO sample metadata
        """
        print("Extracting sample metadata from GEO...")
        
        try:
            # Get sample information from GEO series page
            series_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_series}"
            response = requests.get(series_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find sample links (GSM accessions)
            sample_links = soup.find_all('a', href=lambda href: href and 'acc.cgi?acc=GSM' in href)
            sample_ids = [link.text.strip() for link in sample_links if link.text.strip().startswith('GSM')]
            
            print(f"Found {len(sample_ids)} samples")
            
            if not sample_ids:
                # If no samples found via links, try to find them in the page text
                page_text = soup.get_text()
                import re
                sample_matches = re.findall(r'GSM\d+', page_text)
                sample_ids = list(set(sample_matches))  # Remove duplicates
                print(f"Found {len(sample_ids)} samples via text search")
            
            if not sample_ids:
                print("No samples found - creating series-level record")
                # Create a single row with series information
                result_rows = [{
                    'series_id': geo_series,
                    'series_title': series_metadata.get('title', ''),
                    'series_summary': series_metadata.get('summary', ''),
                    'organism': series_metadata.get('organism', ''),
                    'platform': series_metadata.get('platform', ''),
                    'submission_date': series_metadata.get('submission_date', ''),
                    'sample_id': '',
                    'sample_title': '',
                    'sample_description': '',
                    'data_type': 'GEO_only',
                    'has_sra_data': False
                }]
            else:
                # Extract metadata for each sample and create one row per file
                result_rows = []
                for i, sample_id in enumerate(sample_ids):  # Process all samples
                    print(f"Processing sample {i+1}/{len(sample_ids)}: {sample_id}")
                    
                    sample_metadata = self._get_geo_sample_metadata(sample_id)
                    
                    # Create base row with all sample metadata
                    base_row = {
                        'series_id': geo_series,
                        'series_title': series_metadata.get('title', ''),
                        'series_summary': series_metadata.get('summary', ''),
                        'series_organism': series_metadata.get('organism', ''),
                        'series_platform': series_metadata.get('platform', ''),
                        'series_submission_date': series_metadata.get('submission_date', ''),
                        'sample_id': sample_id,
                        'sample_title': sample_metadata.get('title', ''),
                        'sample_source_name': sample_metadata.get('source_name', ''),
                        'sample_organism': sample_metadata.get('organism', ''),
                        'sample_characteristics': sample_metadata.get('characteristics', ''),
                        'sample_type': sample_metadata.get('sample_type', ''),
                        'extracted_molecule': sample_metadata.get('extracted_molecule', ''),
                        'extraction_protocol': sample_metadata.get('extraction_protocol', ''),
                        'label': sample_metadata.get('label', ''),
                        'label_protocol': sample_metadata.get('label_protocol', ''),
                        'hybridization_protocol': sample_metadata.get('hybridization_protocol', ''),
                        'scan_protocol': sample_metadata.get('scan_protocol', ''),
                        'data_processing': sample_metadata.get('data_processing', ''),
                        'contact_name': sample_metadata.get('contact_name', ''),
                        'contact_email': sample_metadata.get('contact_email', ''),
                        'organization_name': sample_metadata.get('organization_name', ''),
                        'platform_id': sample_metadata.get('platform_id', ''),
                        'data_type': 'GEO_only',
                        'has_sra_data': False
                    }
                    
                    # If files are found, create one row per file
                    if sample_metadata.get('files'):
                        for file_info in sample_metadata['files']:
                            file_row = base_row.copy()
                            file_row.update({
                                'file_name': file_info.get('filename', ''),
                                'file_size': file_info.get('size', ''),
                                'file_type': file_info.get('type', ''),
                                'file_download_url': file_info.get('download_url', '')
                            })
                            result_rows.append(file_row)
                    else:
                        # If no files found, create one row for the sample
                        base_row.update({
                            'file_name': '',
                            'file_size': '',
                            'file_type': '',
                            'file_download_url': ''
                        })
                        result_rows.append(base_row)
                    
                    # Longer delay to avoid rate limiting (NCBI recommends 3 requests per second max)
                    time.sleep(0.5)
            
            # Create DataFrame
            result_df = pd.DataFrame(result_rows)
            
            if output_file:
                result_df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
            
            print(f"Processing complete. Generated {len(result_df)} rows of GEO metadata.")
            return result_df
            
        except Exception as e:
            print(f"Error processing GEO-only series: {e}")
            # Return minimal DataFrame with series info
            result_df = pd.DataFrame([{
                'series_id': geo_series,
                'series_title': series_metadata.get('title', ''),
                'series_summary': series_metadata.get('summary', ''),
                'organism': series_metadata.get('organism', ''),
                'platform': series_metadata.get('platform', ''),
                'data_type': 'GEO_only',
                'has_sra_data': False,
                'error': str(e)
            }])
            
            if output_file:
                result_df.to_csv(output_file, index=False)
                print(f"Minimal results saved to {output_file}")
            
            return result_df
    
    def _get_geo_sample_metadata(self, sample_id: str) -> dict:
        """
        Extract comprehensive metadata for a specific GEO sample using NCBI E-utilities API
        
        Args:
            sample_id: GEO sample identifier (e.g., 'GSM1234567')
            
        Returns:
            Dictionary with comprehensive sample metadata and files
        """
        metadata = {
            'title': '',
            'description': '',
            'organism': '',
            'characteristics': '',
            'source_name': '',
            'sample_type': '',
            'extracted_molecule': '',
            'extraction_protocol': '',
            'label': '',
            'label_protocol': '',
            'hybridization_protocol': '',
            'scan_protocol': '',
            'data_processing': '',
            'submission_date': '',
            'last_update_date': '',
            'contact_name': '',
            'contact_email': '',
            'organization_name': '',
            'platform_id': '',
            'series_id': '',
            'files': []  # List of file dictionaries
        }
        
        try:
            # Go directly to SOFT format - it's more reliable for GEO samples
            self._get_geo_soft_metadata(sample_id, metadata)
                
        except Exception as e:
            print(f"Error extracting sample metadata for {sample_id}: {e}")
            # If SOFT fails, try minimal extraction
            metadata['title'] = sample_id  # At minimum, use the sample ID
            
        return metadata
    
    def _extract_geo_xml_metadata(self, root, metadata: dict, sample_id: str):
        """Extract metadata from GEO XML structure"""
        try:
            # Navigate the XML structure to find sample information
            for sample in root.iter():
                if sample.tag == 'Sample' or 'sample' in sample.tag.lower():
                    # Extract basic fields
                    for child in sample:
                        if 'title' in child.tag.lower():
                            metadata['title'] = child.text or ''
                        elif 'description' in child.tag.lower():
                            metadata['description'] = child.text or ''
                        elif 'organism' in child.tag.lower():
                            metadata['organism'] = child.text or ''
                        elif 'source' in child.tag.lower():
                            metadata['source_name'] = child.text or ''
                        elif 'platform' in child.tag.lower():
                            metadata['platform_id'] = child.text or ''
                        elif 'characteristic' in child.tag.lower():
                            if metadata['characteristics']:
                                metadata['characteristics'] += '; ' + (child.text or '')
                            else:
                                metadata['characteristics'] = child.text or ''
                                
                    # Look for supplementary files
                    for supp_file in sample.iter():
                        if 'supplementary' in supp_file.tag.lower() or 'file' in supp_file.tag.lower():
                            filename = ''
                            filesize = ''
                            filetype = ''
                            
                            for attr in supp_file.attrib:
                                if 'name' in attr.lower():
                                    filename = supp_file.attrib[attr]
                                elif 'size' in attr.lower():
                                    filesize = supp_file.attrib[attr]
                                elif 'type' in attr.lower():
                                    filetype = supp_file.attrib[attr]
                            
                            if filename and '.' in filename:
                                metadata['files'].append({
                                    'filename': filename,
                                    'size': filesize,
                                    'type': filetype,
                                    'download_url': ''
                                })
                                
        except Exception as e:
            print(f"Error parsing XML for {sample_id}: {e}")
    
    def _get_geo_soft_metadata(self, sample_id: str, metadata: dict):
        """Get GEO metadata using SOFT format as fallback"""
        try:
            # Add delay before API call to respect rate limits
            time.sleep(0.4)
            
            # Get SOFT format data
            soft_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sample_id}&targ=self&view=quick&form=text"
            response = requests.get(soft_url, timeout=30)
            response.raise_for_status()
            
            soft_data = response.text
            lines = soft_data.split('\n')
            
            current_section = ''
            for line in lines:
                line = line.strip()
                if line.startswith('!Sample_'):
                    # Parse SOFT format fields
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Map SOFT fields to our metadata structure
                        if key == '!Sample_title':
                            metadata['title'] = value
                        elif key == '!Sample_description':
                            metadata['description'] = value
                        elif key == '!Sample_organism_ch1':
                            metadata['organism'] = value
                        elif key == '!Sample_source_name_ch1':
                            metadata['source_name'] = value
                        elif key == '!Sample_characteristics_ch1':
                            if metadata['characteristics']:
                                metadata['characteristics'] += '; ' + value
                            else:
                                metadata['characteristics'] = value
                        elif key == '!Sample_molecule_ch1':
                            metadata['extracted_molecule'] = value
                        elif key == '!Sample_extract_protocol_ch1':
                            metadata['extraction_protocol'] = value
                        elif key == '!Sample_label_ch1':
                            metadata['label'] = value
                        elif key == '!Sample_label_protocol_ch1':
                            metadata['label_protocol'] = value
                        elif key == '!Sample_hyb_protocol':
                            metadata['hybridization_protocol'] = value
                        elif key == '!Sample_scan_protocol':
                            metadata['scan_protocol'] = value
                        elif key == '!Sample_data_processing':
                            metadata['data_processing'] = value
                        elif key == '!Sample_submission_date':
                            metadata['submission_date'] = value
                        elif key == '!Sample_last_update_date':
                            metadata['last_update_date'] = value
                        elif key == '!Sample_contact_name':
                            metadata['contact_name'] = value
                        elif key == '!Sample_contact_email':
                            metadata['contact_email'] = value
                        elif key == '!Sample_contact_institute':
                            metadata['organization_name'] = value
                        elif key == '!Sample_platform_id':
                            metadata['platform_id'] = value
                        elif key == '!Sample_series_id':
                            metadata['series_id'] = value
                        elif key == '!Sample_supplementary_file':
                            # Parse supplementary file information
                            if value and '.' in value:
                                # Extract filename from the file path/URL
                                filename = value.split('/')[-1] if '/' in value else value
                                metadata['files'].append({
                                    'filename': filename,
                                    'size': '',
                                    'type': '',
                                    'download_url': value if value.startswith('http') else ''
                                })
                                
        except Exception as e:
            print(f"Error getting SOFT data for {sample_id}: {e}")
    
    def _combine_metadata(self, series_metadata: Dict, sra_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Combine series and SRA metadata
        
        Args:
            series_metadata: Dictionary of series-level metadata
            sra_metadata: DataFrame of SRA metadata
            
        Returns:
            Combined DataFrame
        """
        # Start with SRA metadata as the base
        combined_df = sra_metadata.copy()
        
        # Add series-level metadata with 'geo_' prefix
        for key, value in series_metadata.items():
            combined_df[f'geo_{key}'] = value
        
        # Reorder columns for better readability
        column_order = self._get_preferred_column_order(combined_df.columns)
        combined_df = combined_df.reindex(columns=column_order)
        
        return combined_df
    
    def _add_file_information(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add file download URLs and metadata including original format files
        
        Args:
            df: DataFrame with SRA metadata
            
        Returns:
            DataFrame with additional file information including original files
        """
        if df.empty or 'run_accession' not in df.columns:
            return df
        
        # Get detailed file information for each run
        expanded_rows = []
        
        print("Retrieving detailed file information...")
        for _, row in df.iterrows():
            run_acc = row['run_accession']
            
            # Get file details for this run
            file_details = self._get_run_file_details(run_acc)
            
            if file_details:
                # Create a row for each file
                for file_info in file_details:
                    new_row = row.copy()
                    
                    # Add file-specific information - only direct API values
                    new_row['file_name'] = file_info['filename']
                    new_row['file_type'] = file_info['filetype']
                    new_row['file_format'] = file_info['filetype']  # Use original value, no transformation
                    new_row['file_size_bytes'] = file_info['size']
                    new_row['file_size_mb'] = self._convert_size_to_mb(file_info['size'])
                    new_row['file_download_url'] = file_info.get('url', '')
                    new_row['file_s3_location'] = file_info.get('s3_location', '')
                    new_row['file_md5'] = file_info.get('md5', '')
                    
                    expanded_rows.append(new_row)
            else:
                # No detailed file info available from API - skip this run
                print(f"  Warning: No file details available for {run_acc}, skipping")
                continue
        
        if not expanded_rows:
            return df
        
        # Create new DataFrame with expanded rows
        result_df = pd.DataFrame(expanded_rows)
        
        # Remove old generic URL columns if they exist
        columns_to_remove = ['sra_download_url', 'fastq_download_url', 'size_mb']
        for col in columns_to_remove:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])
        
        return result_df
    
    def _get_run_file_details(self, run_accession: str) -> List[Dict]:
        """
        Get detailed file information for a specific SRA run including original format files
        
        Args:
            run_accession: SRA run accession (e.g., 'SRR21492342')
            
        Returns:
            List of dictionaries containing file details
        """
        import requests
        import xml.etree.ElementTree as ET
        import time
        
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            # Use NCBI's E-utilities to get detailed run information
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'sra',
                'term': f'{run_accession}[Accession]',
                'retmode': 'xml'
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=30)
            search_response.raise_for_status()
            
            search_root = ET.fromstring(search_response.content)
            id_list = search_root.find('IdList')
            
            if id_list is None or len(id_list) == 0:
                return []
            
            sra_id = id_list[0].text
            
            # Fetch detailed XML
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                'db': 'sra',
                'id': sra_id,
                'retmode': 'xml'
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            fetch_response.raise_for_status()
            
            # Parse XML to extract file information
            root = ET.fromstring(fetch_response.content)
            files = []
            
            # Look for RUN_DATA_FILE elements which contain file details
            for run_data_file in root.findall('.//RUN_DATA_FILE'):
                file_info = {}
                
                # Get filename
                filename_elem = run_data_file.get('filename')
                if filename_elem:
                    file_info['filename'] = filename_elem
                else:
                    continue
                
                # Get file type
                filetype_elem = run_data_file.get('filetype')
                file_info['filetype'] = filetype_elem or ''
                
                # Get file size
                size_elem = run_data_file.get('size')
                file_info['size'] = size_elem or ''
                
                # Get MD5
                md5_elem = run_data_file.get('md5')
                file_info['md5'] = md5_elem or ''
                
                # Initialize URL and S3 location fields
                file_info['url'] = ''
                file_info['s3_location'] = ''
                
                # Only use direct API values - no construction
                if 'filename' in file_info:
                    filename = file_info['filename']
                    
                    # Only accept direct cloud locations from API
                    if filename.startswith('s3://'):
                        file_info['s3_location'] = filename
                        # Convert s3:// to https URL only for known patterns
                        if filename.startswith('s3://sra-pub-run-odp/'):
                            file_info['url'] = filename.replace('s3://sra-pub-run-odp/', 'https://sra-pub-run-odp.s3.amazonaws.com/')
                        elif filename.startswith('s3://sra-pub-src-'):
                            file_info['url'] = filename.replace('s3://sra-pub-src-', 'https://sra-pub-src-')
                        else:
                            file_info['url'] = filename
                    elif filename.startswith('gs://'):
                        file_info['s3_location'] = filename
                        file_info['url'] = filename  # Keep as-is for GCP
                    elif filename.startswith('https://'):
                        # Direct HTTPS URL provided by API
                        file_info['url'] = filename
                        file_info['s3_location'] = ''  # No S3 location for HTTPS URLs
                    else:
                        # Relative path - leave S3 location and URL blank (no construction)
                        file_info['s3_location'] = ''
                        file_info['url'] = ''
                
                files.append(file_info)
            
            # If no files found in RUN_DATA_FILE, look for alternative structures
            if not files:
                # Look for SRAFile elements (alternative structure)
                for sra_file in root.findall('.//SRAFile'):
                    file_info = {}
                    
                    # Get attributes - only use what's directly provided by API
                    filename = sra_file.get('filename')
                    if filename:  # Only proceed if filename is provided by API
                        file_info['filename'] = filename
                        file_info['filetype'] = sra_file.get('filetype', '')  # Empty if not provided
                        file_info['size'] = sra_file.get('size', '')
                        file_info['md5'] = sra_file.get('md5', '')
                        
                        # Initialize URL and S3 location fields
                        file_info['url'] = ''
                        file_info['s3_location'] = ''
                        
                        # Only use direct locations - no construction
                        if filename.startswith('s3://'):
                            file_info['s3_location'] = filename
                            if filename.startswith('s3://sra-pub-run-odp/'):
                                file_info['url'] = filename.replace('s3://sra-pub-run-odp/', 'https://sra-pub-run-odp.s3.amazonaws.com/')
                            elif filename.startswith('s3://sra-pub-src-'):
                                file_info['url'] = filename.replace('s3://sra-pub-src-', 'https://sra-pub-src-')
                            else:
                                file_info['url'] = filename
                        elif filename.startswith('gs://'):
                            file_info['s3_location'] = filename
                            file_info['url'] = filename
                        elif filename.startswith('https://'):
                            file_info['url'] = filename
                            file_info['s3_location'] = ''
                        else:
                            # Relative path - leave blank
                            file_info['s3_location'] = ''
                            file_info['url'] = ''
                        
                        files.append(file_info)
            
            return files
            
        except Exception as e:
            print(f"Warning: Could not get detailed file info for {run_accession}: {e}")
            return []
    
    def _convert_size_to_mb(self, size_str) -> str:
        """Convert size string to MB"""
        try:
            if pd.isna(size_str) or size_str == '':
                return ''
            
            size_bytes = int(size_str)
            size_mb = round(size_bytes / (1024 * 1024), 2)
            return str(size_mb)
        except (ValueError, TypeError):
            return str(size_str)
    
    def _get_preferred_column_order(self, columns: List[str]) -> List[str]:
        """
        Define a preferred order for columns in the output
        
        Args:
            columns: List of all available columns
            
        Returns:
            Reordered list of columns
        """
        # Define preferred order groups
        priority_columns = [
            # File identifiers
            'run_accession',
            'file_name',
            'file_type',
            'file_format',
            
            # Core identifiers
            'sample_accession',
            'experiment_accession',
            'study_accession',
            
            # GEO series information
            'geo_series_id',
            'geo_title',
            
            # Study/experiment titles
            'study_title',
            'experiment_title',
            'sample_title',
            
            # Biological information
            'organism_name',
            'geo_organism',
            'organism_taxid',
            'library_strategy',
            'library_source',
            'library_selection',
            'library_layout',
            'geo_platform',
            
            # Technical details
            'instrument',
            'instrument_model',
            'bioproject',
            'biosample',
            
            # File information - NEW detailed file info
            'file_size_bytes',
            'file_size_mb',
            'file_download_url',
            'file_s3_location',
            'file_md5',
            
            # Run-level information
            'total_spots',
            'total_size',
            'run_total_spots',
            'run_total_bases',
            
            # Dates
            'geo_submission_date',
            'geo_last_update_date',
        ]
        
        # Start with priority columns that exist
        ordered_columns = [col for col in priority_columns if col in columns]
        
        # Add remaining columns
        remaining_columns = [col for col in columns if col not in ordered_columns]
        ordered_columns.extend(sorted(remaining_columns))
        
        return ordered_columns
    
    def get_series_summary(self, geo_series: str) -> Dict:
        """
        Get a summary of a GEO series without processing all files
        
        Args:
            geo_series: GEO series identifier
            
        Returns:
            Dictionary with series summary information
        """
        try:
            # Get series metadata
            series_metadata = self._get_geo_series_metadata(geo_series)
            
            # Try to map to SRA
            try:
                srp_df = self.sraweb.gse_to_srp(geo_series)
                if not srp_df.empty:
                    srp_id = srp_df['study_accession'].iloc[0]
                    
                    # Get count of SRA records
                    try:
                        sra_metadata = self.sraweb.sra_metadata(srp_id)
                        total_sra_runs = len(sra_metadata)
                        
                        # Get unique samples and experiments
                        unique_samples = sra_metadata['sample_accession'].nunique() if 'sample_accession' in sra_metadata.columns else 0
                        unique_experiments = sra_metadata['experiment_accession'].nunique() if 'experiment_accession' in sra_metadata.columns else 0
                        
                        # Get organisms
                        organisms = sra_metadata['organism_name'].unique().tolist() if 'organism_name' in sra_metadata.columns else []
                        
                        # Get library strategies
                        strategies = sra_metadata['library_strategy'].unique().tolist() if 'library_strategy' in sra_metadata.columns else []
                        
                    except Exception:
                        total_sra_runs = 0
                        unique_samples = 0
                        unique_experiments = 0
                        organisms = []
                        strategies = []
                else:
                    srp_id = None
                    total_sra_runs = 0
                    unique_samples = 0
                    unique_experiments = 0
                    organisms = []
                    strategies = []
                    
            except Exception:
                srp_id = None
                total_sra_runs = 0
                unique_samples = 0
                unique_experiments = 0
                organisms = []
                strategies = []
            
            summary = {
                'series_id': series_metadata.get('series_id', geo_series),
                'title': series_metadata.get('title', ''),
                'organism': series_metadata.get('organism', ''),
                'platform': series_metadata.get('platform', ''),
                'submission_date': series_metadata.get('submission_date', ''),
                'sra_project': srp_id,
                'total_sra_runs': total_sra_runs,
                'unique_samples': unique_samples,
                'unique_experiments': unique_experiments,
                'organisms_in_sra': organisms,
                'library_strategies': strategies,
                'has_sra_data': total_sra_runs > 0
            }
            
            return summary
            
        except Exception as e:
            return {
                'error': f"Failed to get summary for {geo_series}: {str(e)}"
            }