import torch
from safetensors.torch import save_file
from io import BytesIO
import torch.serialization

def convert_pt_to_safetensors(pt_file_path, output_file_path=None):
    """
    将PyTorch .pt权重文件转换为.safetensors格式
    
    参数:
        pt_file_path (str): 输入的.pt文件路径
        output_file_path (str, optional): 输出的.safetensors文件路径。如果为None，则使用与输入相同的名称，但扩展名为.safetensors
    
    返回:
        str: 保存的safetensors文件的路径
    """
    # 设置默认输出路径
    if output_file_path is None:
        output_file_path = pt_file_path.replace('.pt', '.safetensors')
        if output_file_path == pt_file_path:  # 如果文件没有.pt扩展名
            output_file_path = pt_file_path + '.safetensors'
    
    # 添加BytesIO到安全全局变量列表
    torch.serialization.add_safe_globals([BytesIO])
    
    # 加载PyTorch权重
    try:
        state_dict = torch.load(pt_file_path, map_location="cpu")
        print(f"成功加载模型权重文件: {pt_file_path}")
    except Exception as e:
        print(f"加载时出错: {str(e)}")
        print("尝试使用weights_only=False模式...")
        # 备选方案：禁用安全模式
        state_dict = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        print("成功使用非安全模式加载模型")
    
    # 有时权重可能嵌套在'model'或'state_dict'键下
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # 处理复数类型的张量
    processed_state_dict = {}
    for key, tensor in state_dict.items():
        # 跳过非张量类型
        if not isinstance(tensor, torch.Tensor):
            print(f"跳过非张量数据: {key}, 类型: {type(tensor)}")
            continue
            
        # 如果是复数类型，分解为实部和虚部，并创建独立副本
        if tensor.is_complex():
            print(f"处理复数张量: {key}")
            processed_state_dict[f"{key}.real"] = tensor.real.clone()  # 创建独立副本
            processed_state_dict[f"{key}.imag"] = tensor.imag.clone()  # 创建独立副本
        else:
            processed_state_dict[key] = tensor
    
    # 保存为safetensors格式
    try:
        save_file(processed_state_dict, output_file_path)
        print(f"已将模型权重从 {pt_file_path} 转换并保存至 {output_file_path}")
    except Exception as e:
        print(f"保存时出错: {str(e)}")
        # 打印有问题的张量信息以便调试
        for key, tensor in processed_state_dict.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    return output_file_path

# 使用示例
if __name__ == "__main__":
    # 示例用法
    pt_path = "/mbz/users/haolong.jia/opt/torchtitan/c.pt"
    convert_pt_to_safetensors(pt_path, output_file_path="/mbz/users/haolong.jia/opt/torchtitan/c.safetensors")